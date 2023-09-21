# Python Virtual Environment Activation Scripts

This README provides instructions for activating a Python virtual environment using Bash and PowerShell scripts on macOS, Windows, and Linux.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Bash Script](#bash-script)
  - [Linux and macOS](#linux-and-macos)
- [PowerShell Script](#powershell-script)
  - [Windows and macOS](#windows-and-macos)
- [Usage](#usage)

## Prerequisites

- Python 3.x installed
- Bash shell (for Linux and macOS)
- PowerShell (for Windows and optionally macOS)

## Bash Script

The Bash script is compatible with Linux and macOS systems that have a Bash shell.

### Linux and macOS

1. **Save the Script**: Save the following Bash script to a file, for example, `create_venv.sh`.

    ```bash
    #!/bin/bash

    # Using the system python3 to create a new venv
    python3 -m venv week_3_venv

    # Source the activate script to activate the venv
    source week_3_venv/bin/activate
    ```

2. **Make it Executable**: Open a terminal and navigate to the directory where the script is saved. Run the following command to make the script executable.

    ```bash
    chmod +x create_venv.sh
    ```

3. **Run the Script**: Execute the script to create and activate the virtual environment.

    ```bash
    ./create_venv.sh
    ```

## PowerShell Script

The PowerShell script is compatible with Windows and can also be used on macOS if PowerShell is installed.

### Windows and macOS

1. **Save the Script**: Save the following PowerShell script to a file with a `.ps1` extension, for example, `create_venv.ps1`.

    ```bash
    # Using the system python3 to create a new venv
    python3 -m venv week_3_venv

    # Activate the virtual environment
    . .\week_3_venv\Scripts\Activate
    ```

2. **Run the Script**: Open PowerShell and navigate to the directory where the script is saved. Run the script to create and activate the virtual environment.

    ```powershell
    .\create_venv.ps1
    ```

## Usage

- For Linux and macOS, use the Bash script.
- For Windows, use the PowerShell script.
- Optionally, macOS users can also use the PowerShell script if PowerShell is installed.

---
