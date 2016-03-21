# Copyright (C) 2015-2016: The University of Edinburgh
#                 Authors: Craig Warren and Antonis Giannopoulos
#
# This file is part of gprMax.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax.  If not, see <http://www.gnu.org/licenses/>.

"""gprMax.gprMax: provides entry point main()."""

import argparse, datetime, itertools, os, psutil, sys
from time import perf_counter
from enum import Enum

import numpy as np

import gprMax
from gprMax.constants import c, e0, m0, z0, floattype
from gprMax.exceptions import GeneralError
from gprMax.fields_update import update_electric, update_magnetic, update_electric_dispersive_multipole_A, update_electric_dispersive_multipole_B, update_electric_dispersive_1pole_A, update_electric_dispersive_1pole_B
from gprMax.grid import FDTDGrid, dispersion_check
from gprMax.input_cmds_geometry import process_geometrycmds
from gprMax.input_cmds_file import python_code_blocks, write_python_processed, check_cmd_names, compile_user_input_file
from gprMax.input_cmds_multiuse import process_multicmds
from gprMax.input_cmds_singleuse import process_singlecmds
from gprMax.materials import Material
from gprMax.writer_hdf5 import prepare_hdf5, write_hdf5
from gprMax.pml import build_pmls, update_electric_pml, update_magnetic_pml
from gprMax.utilities import update_progress, logo, human_size
from gprMax.yee_cell_build import build_electric_components, build_magnetic_components
from ._version import __version__

def run(**kwargs):
    """
        Function to run gprMax programmatically if you have installed
        it as a module

        n (int): Total number of model runs.
        mpi (bool)   : Switch on MPI Task Farm
        benchmark (bool): Switch on benchmarking mode
        geometry_only (bool): Only build model and produce geometry(files)
        write_python (bool): write an input file after any Python code blocks in the original input file have been processed
        opt_taguchi (bool): optimise parameters using the Taguchi optimisation method
    """

    # Set default values which are truthy.
    kwargs['n'] = kwargs.get('n', 1)
    kwargs['input'] = kwargs['scene'].to_commands()
    run_main(kwargs)

def main():

    """CLI interface for GPRMax"""

    # Parse command line arguments
    parser = argparse.ArgumentParser(prog='gprMax', description='Electromagnetic modelling software based on the Finite-Difference Time-Domain (FDTD) method')
    parser.add_argument('inputfile', help='path to and name of inputfile')
    parser.add_argument('-n', default=1, type=int, help='number of times to run the input file')
    parser.add_argument('-mpi', action='store_true', default=False, help='switch on MPI task farm')
    parser.add_argument('-benchmark', action='store_true', default=False, help='switch on benchmarking mode')
    parser.add_argument('--geometry-only', action='store_true', default=False, help='only build model and produce geometry file(s)')
    parser.add_argument('--write-python', action='store_true', default=False, help='write an input file after any Python code blocks in the original input file have been processed')
    parser.add_argument('--opt-taguchi', action='store_true', default=False, help='optimise parameters using the Taguchi optimisation method')
    args = parser.parse_args()

    # Print gprMax logo, version, and licencing/copyright information
    logo(__version__ + ' (Bowmore)')

    # Cast namespace object as a dict
    model_params = vars(args)

    input_file = args.inputfile

    # Process the input file paths
    model_params['input_directory'] = os.path.dirname(os.path.abspath(input_file)) + os.sep
    model_params['inputfile'] = model_params['input_directory'] + os.path.basename(input_file)

    compiled = compile_user_input_file(model_params)

    model_params['input'] = compiled

    run_main(model_params)

def run_main(model_params):

    """
    Main Function of GPRMax. CLI and module interface both call this
    function

    Args:
        args (dict): Namespace with command line arguments
        numbermodelruns (int): Total number of model runs.
        inputfile (str): Name of the input file to open.
        usernamespace (dict): Namespace that can be accessed by user in any Python code blocks in input file.
        optparams (dict): Optional argument. For Taguchi optimisation it provides the parameters to optimise and their values.
    """

    n = model_params['n']

    # Process for Taguchi optimisation
    if model_params.get('opt_taguchi'):
        if model_params.get('benchmark'):
            raise GeneralError('Taguchi optimisation should not be used with benchmarking mode')
        from gprMax.optimisation_taguchi import run_opt_sim
        run_opt_sim(model_params)

    # Process for benchmarking simulation
    elif model_params.get('benchmark'):
        run_benchmark_sim(model_params)

    # Process for standard simulation
    else:
        # Mixed mode MPI/OpenMP - MPI task farm for models with each model parallelised with OpenMP
        if model_params.get('mpi'):
            if model_params.get('benchmark'):
                raise GeneralError('MPI should not be used with benchmarking mode')
            if n == 1:
                raise GeneralError('MPI is not beneficial when there is only one model to run')
            run_mpi_sim(model_params)
        # Standard behaviour - models run serially with each model parallelised with OpenMP
        else:
            run_std_sim(model_params)

        print('\nSimulation completed.\n{}\n'.format(68*'*'))


def run_std_sim(model_params, optparams=None):

    """Run standard simulation - models are run one after another and each model is parallelised with OpenMP

    Args:
        args (dict): Namespace with command line arguments
        numbermodelruns (int): Total number of model runs.
        inputfile (str): Name of the input file to open.
        usernamespace (dict): Namespace that can be accessed by user in any Python code blocks in input file.
        optparams (dict): Optional argument. For Taguchi optimisation it provides the parameters to optimise and their values.
    """

    tsimstart = perf_counter()
    for model_run in range(1, model_params['n'] + 1):
        model_params['model_run'] = model_run
        run_model(model_params)
    tsimend = perf_counter()
    print('\nTotal simulation time [HH:MM:SS]: {}'.format(datetime.timedelta(seconds=int(tsimend - tsimstart))))


def run_benchmark_sim(model_params):
    """Run standard simulation in benchmarking mode - models are run one after another and each model is parallelised with OpenMP

    Args:
        args (dict): Namespace with command line arguments
        inputfile (str): Name of the input file to open.
        usernamespace (dict): Namespace that can be accessed by user in any Python code blocks in input file.
    """

    number_model_runs = model_params['n']

    # Number of threads to test - start from max physical CPU cores and divide in half until 1
    thread = psutil.cpu_count(logical=False)
    threads = [thread]
    while not thread%2:
        thread /= 2
        threads.append(int(thread))

    benchtimes = np.zeros(len(threads))

    numbermodelruns = len(threads)
    tsimstart = perf_counter()
    for model_run in range(1, number_model_runs + 1):
        model_params['model_run'] = model_run
        os.environ['OMP_NUM_THREADS'] = str(threads[model_run - 1])
        tsolve = run_model(model_params)
        benchtimes[model_run - 1] = tsolve
    tsimend = perf_counter()

    # Save number of threads and benchmarking times to NumPy archive
    threads = np.array(threads)
    np.savez(os.path.splitext(model_params['inputfile'])[0], threads=threads, benchtimes=benchtimes)

    print('\nTotal simulation time [HH:MM:SS]: {}'.format(datetime.timedelta(seconds=int(tsimend - tsimstart))))


def run_mpi_sim(model_params, optparams=None):
    """Run mixed mode MPI/OpenMP simulation - MPI task farm for models with each model parallelised with OpenMP

    Args:
        args (dict): Namespace with command line arguments
        numbermodelruns (int): Total number of model runs.
        inputfile (str): Name of the input file to open.
        usernamespace (dict): Namespace that can be accessed by user in any Python code blocks in input file.
        optparams (dict): Optional argument. For Taguchi optimisation it provides the parameters to optimise and their values.
    """

    from mpi4py import MPI

    number_model_runs = model_params['n']

    # Define MPI message tags
    tags = Enum('tags', {'READY': 0, 'DONE': 1, 'EXIT': 2, 'START': 3})

    # Initializations and preliminaries
    comm = MPI.COMM_WORLD   # get MPI communicator object
    size = comm.size        # total number of processes
    rank = comm.rank        # rank of this process
    status = MPI.Status()   # get MPI status object
    name = MPI.Get_processor_name()     # get name of processor/host

    if rank == 0: # Master process
        modelrun = 1
        numworkers = size - 1
        closedworkers = 0
        print('Master: PID {} on {} using {} workers.'.format(os.getpid(), name, numworkers))
        while closedworkers < numworkers:
            data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            source = status.Get_source()
            tag = status.Get_tag()

            if tag == tags.READY.value: # Worker is ready, so send it a task
                if model_run < numbermodel_runs + 1:
                    comm.send(model_run, dest=source, tag=tags.START.value)
                    print('Master: sending model {} to worker {}.'.format(model_run, source))
                    model_run += 1
                else:
                    comm.send(None, dest=source, tag=tags.EXIT.value)

            elif tag == tags.DONE.value:
                print('Worker {}: completed.'.format(source))

            elif tag == tags.EXIT.value:
                print('Worker {}: exited.'.format(source))
                closedworkers += 1

    else: # Worker process
        print('Worker {}: PID {} on {} requesting {} OpenMP threads.'.format(rank, os.getpid(), name, os.environ.get('OMP_NUM_THREADS')))
        while True:
            comm.send(None, dest=0, tag=tags.READY.value)
            model_run = comm.recv(source=0, tag=MPI.ANY_TAG, status=status) # Receive a model number to run from the master
            model_params['model_run'] = model_run
            tag = status.Get_tag()

            # Run a model
            if tag == tags.START.value:
                run_model(model_params)
                comm.send(None, dest=0, tag=tags.DONE.value)

            elif tag == tags.EXIT.value:
                break

        comm.send(None, dest=0, tag=tags.EXIT.value)


def run_model(model_params):
    """Runs a model - processes the input file; builds the Yee cells; calculates update coefficients; runs main FDTD loop.

    Args:
        args (dict): Namespace with command line arguments
        modelrun (int): Current model run number.
        numbermodelruns (int): Total number of model runs.
        inputfile (str): Name of the input file to open.
        usernamespace (dict): Namespace that can be accessed by user in any Python code blocks in input file.

    Returns:
        tsolve (int): Length of time (seconds) of main FDTD calculations
    """

    model_run = model_params['model_run']
    number_model_runs = model_params['n']

    processedlines = model_params['input'].split('\n')[:-1]

    # Monitor memory usage
    p = psutil.Process()

    print('\n{}\n\nModel input file: {}\n'.format(68*'*', processedlines))

    # Initialise an instance of the FDTDGrid class
    G = FDTDGrid()
    G.inputdirectory = model_params['input_directory']

    singlecmds, multicmds, geometry = check_cmd_names(processedlines)

    # Process parameters for commands that can only occur once in the model
    process_singlecmds(singlecmds, G)

    # Process parameters for commands that can occur multiple times in the model
    process_multicmds(multicmds, G)

    # Initialise an array for volumetric material IDs (solid), boolean arrays for specifying materials not to be averaged (rigid),
    # an array for cell edge IDs (ID), and arrays for the field components.
    G.initialise_std_arrays()

    # Process the geometry commands in the order they were given
    tinputprocstart = perf_counter()
    process_geometrycmds(geometry, G)
    tinputprocend = perf_counter()
    print('\nInput file processed in [HH:MM:SS]: {}'.format(datetime.timedelta(seconds=int(tinputprocend - tinputprocstart))))

    # Build the PML and calculate initial coefficients
    build_pmls(G)

    # Build the model, i.e. set the material properties (ID) for every edge of every Yee cell
    tbuildstart = perf_counter()
    build_electric_components(G.solid, G.rigidE, G.ID, G)
    build_magnetic_components(G.solid, G.rigidH, G.ID, G)
    tbuildend = perf_counter()
    print('\nModel built in [HH:MM:SS]: {}'.format(datetime.timedelta(seconds=int(tbuildend - tbuildstart))))

    # Process any voltage sources (that have resistance) to create a new material at the source location
    for voltagesource in G.voltagesources:
        voltagesource.create_material(G)

    # Initialise arrays of update coefficients to pass to update functions
    G.initialise_std_updatecoeff_arrays()

    # Initialise arrays of update coefficients and temporary values if there are any dispersive materials
    if Material.maxpoles != 0:
        G.initialise_dispersive_arrays()

    # Calculate update coefficients, store in arrays, and list materials in model
    if G.messages:
        print('\nMaterials:\n')
        print('ID\tName\t\tProperties')
        print('{}'.format('-'*50))
    for material in G.materials:

        # Calculate update coefficients for material
        material.calculate_update_coeffsE(G)
        material.calculate_update_coeffsH(G)

        # Store all update coefficients together
        G.updatecoeffsE[material.numID, :] = material.CA, material.CBx, material.CBy, material.CBz, material.srce
        G.updatecoeffsH[material.numID, :] = material.DA, material.DBx, material.DBy, material.DBz, material.srcm

        # Store coefficients for any dispersive materials
        if Material.maxpoles != 0:
            z = 0
            for pole in range(Material.maxpoles):
                G.updatecoeffsdispersive[material.numID, z:z+3] = e0 * material.eqt2[pole], material.eqt[pole], material.zt[pole]
                z += 3

        if G.messages:
            if material.deltaer and material.tau:
                tmp = 'delta_epsr={}, tau={} secs; '.format(', '.join('{:g}'.format(deltaer) for deltaer in material.deltaer), ', '.join('{:g}'.format(tau) for tau in material.tau))
            else:
                tmp = ''
            if material.average:
                dielectricsmoothing = 'dielectric smoothing permitted.'
            else:
                dielectricsmoothing = 'dielectric smoothing not permitted.'
            print('{:3}\t{:12}\tepsr={:g}, sig={:g} S/m; mur={:g}, sig*={:g} S/m; '.format(material.numID, material.ID, material.er, material.se, material.mr, material.sm) + tmp + dielectricsmoothing)

    # Check to see if numerical dispersion might be a problem
    if dispersion_check(G.waveforms, G.materials, G.dx, G.dy, G.dz):
        print('\nWARNING: Potential numerical dispersion in the simulation. Check the spatial discretisation against the smallest wavelength present.')

    # Write files for any geometry views
    if not G.geometryviews and args.geometry_only:
        raise GeneralError('No geometry views found.')
    elif G.geometryviews:
        tgeostart = perf_counter()
        for geometryview in G.geometryviews:
            geometryview.write_vtk(model_run, number_model_runs, G)
        tgeoend = perf_counter()
        print('\nGeometry file(s) written in [HH:MM:SS]: {}'.format(datetime.timedelta(seconds=int(tgeoend - tgeostart))))

    # Run simulation if not doing only geometry
    if not model_params.get('geometry_only'):

        # Prepare any snapshot files
        for snapshot in G.snapshots:
            snapshot.prepare_vtk_imagedata(model_run, number_model_runs, G)

        # Prepare output file
        for l in processedlines:
            print(l)
        inputfileparts = os.path.splitext(model_params['inputfile'])
        if number_model_runs == 1:
            outputfile = inputfileparts[0] + '.out'
        else:
            outputfile = inputfileparts[0] + str(model_run) + '.out'
        sys.stdout.write('\nOutput to file: {}\n'.format(outputfile))
        sys.stdout.flush()
        f = prepare_hdf5(outputfile, G)

        # Adjust position of sources and receivers if required
        if G.srcstepx > 0 or G.srcstepy > 0 or G.srcstepz > 0:
            for source in itertools.chain(G.hertziandipoles, G.magneticdipoles, G.voltagesources, G.transmissionlines):
                source.xcoord += (model_run - 1) * G.srcstepx
                source.ycoord += (model_run - 1) * G.srcstepy
                source.zcoord += (model_run - 1) * G.srcstepz
        if G.rxstepx > 0 or G.rxstepy > 0 or G.rxstepz > 0:
            for receiver in G.rxs:
                receiver.xcoord += (model_run - 1) * G.rxstepx
                receiver.ycoord += (model_run - 1) * G.rxstepy
                receiver.zcoord += (model_run - 1) * G.rxstepz

        ##################################
        #   Main FDTD calculation loop   #
        ##################################
        tsolvestart = perf_counter()
        # Absolute time
        abstime = 0

        for timestep in range(G.iterations):
            if timestep == 0:
                tstepstart = perf_counter()

            # Write field outputs to file
            write_hdf5(f, timestep, G.Ex, G.Ey, G.Ez, G.Hx, G.Hy, G.Hz, G)

            # Write any snapshots to file
            for snapshot in G.snapshots:
                if snapshot.time == timestep + 1:
                    snapshot.write_vtk_imagedata(G.Ex, G.Ey, G.Ez, G.Hx, G.Hy, G.Hz, G)

            # Update electric field components
            if Material.maxpoles == 0: # All materials are non-dispersive so do standard update
                update_electric(G.nx, G.ny, G.nz, G.nthreads, G.updatecoeffsE, G.ID, G.Ex, G.Ey, G.Ez, G.Hx, G.Hy, G.Hz)
            elif Material.maxpoles == 1: # If there are any dispersive materials do 1st part of dispersive update (it is split into two parts as it requires present and updated electric field values).
                update_electric_dispersive_1pole_A(G.nx, G.ny, G.nz, G.nthreads, G.updatecoeffsE, G.updatecoeffsdispersive, G.ID, G.Tx, G.Ty, G.Tz, G.Ex, G.Ey, G.Ez, G.Hx, G.Hy, G.Hz)
            elif Material.maxpoles > 1:
                update_electric_dispersive_multipole_A(G.nx, G.ny, G.nz, G.nthreads, Material.maxpoles, G.updatecoeffsE, G.updatecoeffsdispersive, G.ID, G.Tx, G.Ty, G.Tz, G.Ex, G.Ey, G.Ez, G.Hx, G.Hy, G.Hz)

            # Update electric field components with the PML correction
            update_electric_pml(G)

            # Update electric field components from sources
            for voltagesource in G.voltagesources:
                voltagesource.update_electric(abstime, G.updatecoeffsE, G.ID, G.Ex, G.Ey, G.Ez, G)
            for transmissionline in G.transmissionlines:
                transmissionline.update_electric(abstime, G.Ex, G.Ey, G.Ez, G)
            for hertziandipole in G.hertziandipoles: # Update any Hertzian dipole sources last
                hertziandipole.update_electric(abstime, G.updatecoeffsE, G.ID, G.Ex, G.Ey, G.Ez, G)

            # If there are any dispersive materials do 2nd part of dispersive update (it is split into two parts as it requires present and updated electric field values). Therefore it can only be completely updated after the electric field has been updated by the PML and source updates.
            if Material.maxpoles == 1:
                update_electric_dispersive_1pole_B(G.nx, G.ny, G.nz, G.nthreads, G.updatecoeffsdispersive, G.ID, G.Tx, G.Ty, G.Tz, G.Ex, G.Ey, G.Ez)
            elif Material.maxpoles > 1:
                update_electric_dispersive_multipole_B(G.nx, G.ny, G.nz, G.nthreads, Material.maxpoles, G.updatecoeffsdispersive, G.ID, G.Tx, G.Ty, G.Tz, G.Ex, G.Ey, G.Ez)

            # Increment absolute time value
            abstime += 0.5 * G.dt

            # Update magnetic field components
            update_magnetic(G.nx, G.ny, G.nz, G.nthreads, G.updatecoeffsH, G.ID, G.Ex, G.Ey, G.Ez, G.Hx, G.Hy, G.Hz)

            # Update magnetic field components with the PML correction
            update_magnetic_pml(G)

            # Update magnetic field components from sources
            for transmissionline in G.transmissionlines:
                transmissionline.update_magnetic(abstime, G.Hx, G.Hy, G.Hz, G)
            for magneticdipole in G.magneticdipoles:
                magneticdipole.update_magnetic(abstime, G.updatecoeffsH, G.ID, G.Hx, G.Hy, G.Hz, G)

            # Increment absolute time value
            abstime += 0.5 * G.dt

            # Calculate time for two iterations, used to estimate overall runtime
            if timestep == 1:
                tstepend = perf_counter()
                runtime = datetime.timedelta(seconds=int((tstepend - tstepstart) / 2 * G.iterations))
                sys.stdout.write('Estimated runtime [HH:MM:SS]: {}\n'.format(runtime))
                sys.stdout.write('Solving for model run {} of {}...\n'.format(model_run, number_model_runs))
                sys.stdout.flush()
            elif timestep > 1:
                update_progress((timestep + 1) / G.iterations)

        # Close output file
        f.close()

        tsolveend = perf_counter()
        print('\n\nSolving took [HH:MM:SS]: {}'.format(datetime.timedelta(seconds=int(tsolveend - tsolvestart))))
        print('Peak memory (approx) used: {}'.format(human_size(p.memory_info().rss)))

        return int(tsolveend - tsolvestart)
