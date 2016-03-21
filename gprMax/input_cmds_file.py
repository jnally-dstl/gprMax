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

import os, sys

from gprMax.exceptions import CmdInputError
from gprMax.utilities import ListStream
from gprMax.constants import c, e0, m0, z0, floattype


def python_code_blocks(inputfile, usernamespace):
    """Looks for and processes any Python code found in the input file. It will ignore any lines that are comments, i.e. begin with a double hash (##), and any blank lines. It will also ignore any lines that do not begin with a hash (#) after it has processed Python commands.

    Args:
        inputfile (str): Name of the input file to open.
        usernamespace (dict): Namespace that can be accessed by user in any Python code blocks in input file.

    Returns:
        processedlines (list): Input commands after Python processing.
    """

    with open(inputfile, 'r') as f:
        # Strip out any newline characters and comments that must begin with double hashes
        inputlines = [line.rstrip() for line in f if(not line.startswith('##') and line.rstrip('\n'))]

    # List to hold final processed commands
    processedlines = []

    x = 0
    while(x < len(inputlines)):
        if(inputlines[x].startswith('#python:')):
            # String to hold Python code to be executed
            pythoncode = ''
            x += 1
            while not inputlines[x].startswith('#end_python:'):
                # Add all code in current code block to string
                pythoncode += inputlines[x] + '\n'
                x += 1
                if x == len(inputlines):
                    raise CmdInputError('Cannot find the end of the Python code block, i.e. missing #end_python: command.')
            # Compile code for faster execution
            pythoncompiledcode = compile(pythoncode, '<string>', 'exec')
            # Redirect stdio to a ListStream
            sys.stdout = codeout = ListStream()
            # Execute code block & make available only usernamespace
            exec(pythoncompiledcode, usernamespace)

            # Now strip out any lines that don't begin with a hash command
            codeproc = [line + ('\n') for line in codeout.data if(line.startswith('#'))]

            # Add processed Python code to list
            processedlines.extend(codeproc)

        elif(inputlines[x].startswith('#')):
            # Add gprMax command to list
            inputlines[x] += ('\n')
            processedlines.append(inputlines[x])

        x += 1

    sys.stdout = sys.__stdout__ # Reset stdio

    return processedlines

def compile_user_input_file(model_params):

    input_file = model_params['inputfile']

    # Process the input file
    inputdirectory = os.path.dirname(os.path.abspath(input_file)) + os.sep
    inputfile = inputdirectory + os.path.basename(input_file)

    # Provide a namespace that any scripted elements in the input file
    # can access
    usernamespace = {'c': c, 'e0': e0, 'm0': m0, 'z0': z0,
        'number_model_runs': model_params['n'],
        'inputdirectory': inputdirectory
    }

    # If Taguchi optimistaion, add specific value for each parameter to optimise for each experiment to user accessible namespace
    if model_params.get('optparams'):
        tmp = {}
        tmp.update((key, value[model_params['model_run'] - 1]) for key, value in optparams.items())
        usernamespace.update({'optparams': tmp})

    # Process any user input Python commands
    processedlines = python_code_blocks(inputfile, usernamespace)

    # Write a file containing the input commands after Python blocks have been processed
    if model_params.get('write_python'):
        write_python_processed(inputfile, modelrun, model_params['n'], processedlines)

    compiled = ''.join(processedlines)

    return compiled

def write_python_processed(inputfile, modelrun, numbermodelruns, processedlines):
    """Writes input commands to file after Python processing.

    Args:
        inputfile (str): Name of the input file to open.
        modelrun (int): Current model run number.
        numbermodelruns (int): Total number of model runs.
        processedlines (list): Input commands after Python processing.
    """

    if numbermodelruns == 1:
        processedfile = os.path.splitext(inputfile)[0] + '_proc.in'
    else:
        processedfile = os.path.splitext(inputfile)[0] + str(modelrun) + '_proc.in'

    with open(processedfile, 'w') as f:
        for item in processedlines:
            f.write('{}'.format(item))

    print('Written input commands after Python processing to file: {}\n'.format(processedfile))


def check_cmd_names(processedlines):
    """Checks the validity of commands, i.e. are they gprMax commands, and that all essential commands are present.

    Args:
        processedlines (list): Input commands after Python processing.

    Returns:
        singlecmds (dict): Commands that can only occur once in the model.
        multiplecmds (dict): Commands that can have multiple instances in the model.
        geometry (list): Geometry commands in the model.
    """

    # Dictionaries of available commands
    # Essential commands neccessary to run a gprMax model
    essentialcmds = ['#domain', '#dx_dy_dz', '#time_window']

    # Commands that there should only be one instance of in a model
    singlecmds = dict.fromkeys(['#domain', '#dx_dy_dz', '#time_window', '#title', '#messages', '#num_threads', '#time_step_stability_factor', '#pml_cells', '#excitation_file', '#src_steps', '#rx_steps', '#taguchi', '#end_taguchi'], 'None')

    # Commands that there can be multiple instances of in a model - these will be lists within the dictionary
    multiplecmds = {key: [] for key in ['#geometry_view', '#material', '#soil_peplinski', '#add_dispersion_debye', '#add_dispersion_lorentz', '#add_dispersion_drude', '#waveform', '#voltage_source', '#hertzian_dipole', '#magnetic_dipole', '#transmission_line', '#rx', '#rx_box', '#snapshot', '#pml_cfs']}

    # Geometry object building commands that there can be multiple instances of in a model - these will be lists within the dictionary
    geometrycmds = ['#xdmf_geometry_file', '#edge', '#plate', '#triangle', '#box', '#sphere', '#cylinder', '#cylindrical_sector', '#fractal_box', '#add_surface_roughness', '#add_surface_water', '#add_grass']
    # List to store all geometry object commands in order from input file
    geometry = []

    # Check if command names are valid, if essential commands are present, and add command parameters to appropriate dictionary values or lists
    countessentialcmds = 0
    lindex = 0
    while(lindex < len(processedlines)):
        cmd = processedlines[lindex].split(':')
        cmdname = cmd[0].lower()

        # Check if command name is valid
        if cmdname not in essentialcmds and cmdname not in singlecmds and cmdname not in multiplecmds and cmdname not in geometrycmds:
            raise CmdInputError('Your input file contains the invalid command: ' + cmdname)

        # Count essential commands
        if cmdname in essentialcmds:
            countessentialcmds += 1

        # Assign command parameters as values to dictionary keys
        if cmdname in singlecmds:
            if singlecmds[cmdname] == 'None':
                singlecmds[cmdname] = cmd[1].strip(' \t\n')
            else:
                raise CmdInputError('You can only have one ' + cmdname + ' commmand in your model')

        elif cmdname in multiplecmds:
            multiplecmds[cmdname].append(cmd[1].strip(' \t\n'))

        elif cmdname in geometrycmds:
            geometry.append(processedlines[lindex].strip(' \t\n'))

        lindex += 1

    if (countessentialcmds < len(essentialcmds)):
        raise CmdInputError('Your input file is missing essential gprMax commands required to run a model. Essential commands are: ' + ', '.join(essentialcmds))

    return singlecmds, multiplecmds, geometry

