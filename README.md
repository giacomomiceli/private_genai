# Proof of concept for the RAG project

## Preparatory steps to use this code on Windows 11

### Tools for managing python packages
This project uses poetry for python dependency management. This allows you to create a virtual environment with the packages specified in poetry.lock by running only the one liner `poetry install` in a PowerShell terminal after cloning this repo.

The recommended way to install poetry is to use pipx. The recommended way to install pipx is to use scoop, the command-line installer tool for Windows. All of those tools are user space tools which install software to C:\Users\YourAccount, so use a regular PowerShell terminal for the next step (not running as administrator).

Before installing scoop verify that your PowerShell execution policy does not prevent you from executing scripts. This is necessary to install scoop.

```powershell
Get-ExecutionPolicy
```

If the output is RemoteSigned, you are good to go. Else run the following command to change the execution policy:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Run this to install scoop (it executes a script hosted on the scoop.sh website):
Invoke-RestMethod -Uri https://get.scoop.sh | Invoke-Expression

Once scoop is installed, you can use it to install pipx.
```powershell
scoop install pipx
pipx ensurepath
```

Finally you can use pipx to install poetry.
```powershell
pipx install poetry
```

### Install Dcker desktop
You need docker to run a local instance of Weaviate. You can install docker desktop from the official website at https://www.docker.com/products/docker-desktop/, or if you use chocolatey run `choco install docker-desktop`. Reboot after installation. The Docker GUI open at startup and will ask you if you want to log in with a docker account to enable online features, this is not necessary for the purpose of this project.

## Running the code

Navigate to the clone folder in a PowerShell non-admin terminal.

Execute poetry to set up python packages (this uses poetry.lock)

```powershell
poetry install
```

Separately open an administrator PowerShell and navigate to the clone folder. Since docker-compose.yaml points to the official weaviate image, you just need to start docker to get a Weaviate instance running at http://localhost:8080/

```powershell
docker compose up
```

Finally open the ipynb files in VSCode or any other editor that supports Jupyter notebooks. Make sure to use the virtual environment managed by poetry for this project (in VSCode this is a drop-down menu choice).

```bash
poetry run streamlit run chat.py
```
