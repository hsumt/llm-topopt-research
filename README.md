# llm-topopt-research — Agentic Topology Optimization (ATO)

Agentic Topology Optimization puts language-model agents alongside an engineer across
the whole topology-optimization workflow. That runs from forming the problem
specification, to configuring and running the solver that implements it, to
diagnosing the result afterwards.

## Design

This is the design ATO is being built toward; the repository implements part of it.

The work is divided like this:
- **The agent makes the decisions.** It interprets the engineer's brief, fills the
  specification, and chooses how the codebase should realize it.
- **The codebase provides the means.** It offers modular, configurable building blocks
  for geometry, loads, supports, materials, objectives, constraints and discretization.
  It validates every request and runs it deterministically.
- **Results come only from the solver.**
- **A human keeps authority** over what is solved.

ATO has two halves:

1. **Pre-solve formulation.** An engineer describes a design problem in plain
   language. The system:
   - turns it into an explicit specification;
   - asks for the missing engineering decisions, without inventing physical values;
   - configures the solver from the specification;
   - shows the exact solver inputs for approval before anything runs.
2. **Post-solve specification diagnosis.** A run can converge and still be wrong,
   because the problem it solved was wrong. ATO:
   - analyses the numerical results and the data recorded during the run;
   - proposes edits to the problem;
   - accepts an edit only once a re-solve confirms it;
   - flags any edit that would change the engineer's intent for a human to decide.

The long-term aim is to run any topology-optimization problem an engineer can
describe, not one benchmark.

## Status

What runs today:
- **Pre-solve formulation**, end to end, in a web app: brief, specification, code check
  and model review, questions, approval of the exact solver inputs, run, results.
- **One connected solver:** a 2D plane-stress L-bracket. It minimizes compliance under
  a volume limit and an optional p-norm stress limit, with a level-set method
  (ParaLeSTO on DOLFINx 0.7). It is a port of mfrto's `simplified_2D_holes` with the
  physics unchanged, and it is provisional: vetted 2D and 3D L-bracket solvers will
  replace it. The app reports any other problem as not supported.
- **Model providers:** Anthropic or Azure OpenAI (see [Example `.env`](#example-env)).

Post-solve diagnosis is not connected yet.

## Quick start

Pick your platform. Each section is a complete setup from scratch.

### Linux

1. Install Docker Engine and the Docker Compose plugin, following the guide for your
   distribution.

```text
https://docs.docker.com/engine/install/
```

2. Let your user run Docker without `sudo`, then log out and back in.

```bash
sudo usermod -aG docker $USER
```

3. Check that Docker works.

```bash
docker compose version
```

4. Install VS Code.

```text
https://code.visualstudio.com/
```

5. Get the code. Run this in the folder where you want the project; it creates an
   `llm-topopt-research` folder there.

```bash
git clone -b develop https://github.com/hsumt/llm-topopt-research.git
```

6. Go into the repository folder.

```bash
cd llm-topopt-research
```

7. Open the folder in VS Code.

```bash
code .
```

8. Add the Anthropic API key you were given and the model to use; every model step needs
   both. Without them the app still starts, but only benchmarks run, since they make no
   model calls. Create a file named `.env` in the repository folder.

```bash
nano .env
```

Add these lines with your key. To save and exit nano, press Ctrl+O, then Enter, then
Ctrl+X. The file is gitignored; never commit it. To use Azure OpenAI instead, see
[Example `.env`](#example-env).

```text
ANTHROPIC_API_KEY=your-key-here
ANTHROPIC_MODEL=claude-sonnet-4-6
```

9. Build and start the app. The first build downloads and compiles the solver
   stack, so it takes a while.

```bash
./docker/compose up --build -d
```

10. Open the app in your browser.

```text
http://localhost:8501
```

11. Stop the app when you are done.

```bash
./docker/compose down
```

The image is x86-64 only. On an ARM64 machine, register x86-64 emulation first; see
https://docs.docker.com/build/building/multi-platform/.

### macOS

1. Install Docker Desktop and start it. On Apple Silicon, also open Settings →
   General and turn on "Use Rosetta for x86_64/amd64 emulation on Apple Silicon".
   The image is x86-64 only, so on Apple Silicon it runs under emulation and is
   slower.

```text
https://docs.docker.com/desktop/setup/install/mac-install/
```

2. Check that Docker works.

```bash
docker compose version
```

3. Install VS Code.

```text
https://code.visualstudio.com/
```

Move VS Code into your Applications folder, then add its `code` command to the
terminal.

```bash
cat << EOF >> ~/.zprofile
# Add Visual Studio Code (code)
export PATH="\$PATH:/Applications/Visual Studio Code.app/Contents/Resources/app/bin"
EOF
```

Load the change into the current terminal.

```bash
source ~/.zprofile
```

Check that the `code` command works.

```bash
code --version
```

4. Get the code. Run this in the folder where you want the project; it creates an
   `llm-topopt-research` folder there.

```bash
git clone -b develop https://github.com/hsumt/llm-topopt-research.git
```

5. Go into the repository folder.

```bash
cd llm-topopt-research
```

6. Open the folder in VS Code.

```bash
code .
```

7. Add the Anthropic API key you were given and the model to use; every model step needs
   both. Without them the app still starts, but only benchmarks run, since they make no
   model calls. Create a file named `.env` in the repository folder.

```bash
nano .env
```

Add these lines with your key. To save and exit nano, press Control+O, then Enter, then
Control+X. The file is gitignored; never commit it. To use Azure OpenAI instead, see
[Example `.env`](#example-env).

```text
ANTHROPIC_API_KEY=your-key-here
ANTHROPIC_MODEL=claude-sonnet-4-6
```

8. Point the app at a ParaView folder that Docker Desktop shares. Docker Desktop on
   macOS shares only some folders by default, such as `/Users`, so the default
   `/opt/paraview` can stop the app from starting. An empty folder is fine; see
   [ParaView](#paraview-optional) to fill it.

```bash
mkdir -p ~/paraview && printf '\nPARAVIEW_DIR=%s\n' "$HOME/paraview" >> .env
```

9. Build and start the app. The first build downloads and compiles the solver
   stack, so it takes a while.

```bash
./docker/compose up --build -d
```

10. Open the app in your browser.

```text
http://localhost:8501
```

11. Stop the app when you are done.

```bash
./docker/compose down
```

### Windows

1. Install WSL from PowerShell run as Administrator, then restart.

```powershell
wsl --install --web-download
```

2. Install Ubuntu from PowerShell, and create a Linux username and password when asked.

```powershell
wsl --install -d Ubuntu --web-download
```

3. Make Ubuntu the default. Run this in the Ubuntu terminal that step 2 left open;
   the list should show `*` next to Ubuntu.

```bash
wsl.exe --set-default Ubuntu
wsl.exe -l -v
```

4. Install Docker Desktop and start it, then turn on Ubuntu: Settings → Resources →
   WSL integration → Ubuntu toggle → Apply.

```text
https://docs.docker.com/desktop/setup/install/windows-install/
```

5. Install VS Code.

```text
https://code.visualstudio.com/
```

6. Install Microsoft's **WSL** extension in VS Code: Extensions → search `WSL` →
   Install, or run this in PowerShell.

```powershell
code --install-extension ms-vscode-remote.remote-wsl
```

7. Open VS Code and an Ubuntu terminal. Run every remaining command there.

```text
Terminal → New Terminal → + → Ubuntu (WSL)
```

8. Check that Docker works.

```bash
docker compose version
```

9. Go to your Linux home folder; cloning under `/mnt/c` is slow and can break file
   permissions.

```bash
cd ~
```

10. Clone the `develop` branch into your Linux home; this creates an
   `llm-topopt-research` folder.

```bash
git clone -b develop https://github.com/hsumt/llm-topopt-research.git
```

11. Go into the repository folder.

```bash
cd llm-topopt-research
```

12. Open the folder in VS Code from the Ubuntu terminal. The bottom-left corner should
   show **WSL: Ubuntu**.

```bash
code .
```

13. Add the Anthropic API key you were given and the model to use; every model step needs
   both. Without them the app still starts, but only benchmarks run, since they make no
   model calls. Create a file named `.env` in the repository folder.

```bash
nano .env
```

Add these lines with your key. To save and exit nano, press Ctrl+O, then Enter, then
Ctrl+X. The file is gitignored; never commit it. To use Azure OpenAI instead, see
[Example `.env`](#example-env).

```text
ANTHROPIC_API_KEY=your-key-here
ANTHROPIC_MODEL=claude-sonnet-4-6
```

14. Build and start the app. The first build downloads and compiles the solver
   stack, so it takes a while.

```bash
./docker/compose up --build -d
```

15. Open the app in your Windows browser.

```text
http://localhost:8501
```

16. Stop the app when you are done.

```bash
./docker/compose down
```

## ParaView (optional)

Each run can render its result with ParaView: one cube per element, in seven standard
views. The app uses a ParaView install on your computer, mounted read-only into the
container. Without one, runs still complete; the run's page shows the app's own element
view and says why there is no ParaView render.

1. Download the ParaView binary for Linux x86-64 (this repository uses 6.1). On macOS
   and Windows too, use the Linux build: it runs inside the Linux container.

```text
https://www.paraview.org/download/
```

2. Extract it so that `bin/pvbatch` sits directly in the mounted folder. The commands
   read the archive from `~/Downloads`; change the path if it was saved elsewhere. In
   WSL, a file downloaded in Windows is under `/mnt/c/Users/<your Windows user name>/Downloads/`.
   On Linux and in WSL the default folder is `/opt/paraview`:

```bash
sudo mkdir -p /opt/paraview && sudo tar -xzf ~/Downloads/ParaView-*-Linux-*.tar.gz -C /opt/paraview --strip-components=1
```

   On macOS, extract into the folder that macOS step 8 set as `PARAVIEW_DIR`:

```bash
tar -xzf ~/Downloads/ParaView-*-Linux-*.tar.gz -C ~/paraview --strip-components=1
```

3. Recreate the container so it picks up the install. To use another folder, set
   `PARAVIEW_DIR` to it in `.env` first.

```bash
./docker/compose up -d --force-recreate
```

## Using the app

The sidebar has these pages:
- **Home:** where to start, and how many formulations are in each state.
- **New problem:** describe a design problem in plain language, or start from a
  suggested problem. The benchmark is one of them: its documented inputs as a brief
  you can change. The system reads the brief into a specification, checks it against
  the connected solver, has a model review it, and asks for the decisions that are
  missing.
- **Benchmarks:** run a documented case exactly as published, straight to approval with
  no model calls, or open it as a new problem to change it first.
- **Batch:** approve several ready formulations together.
- **Jobs:** every formulation with its state, and every approval.
- **Process flow:** how a formulation was derived, step by step, with the code and
  prompts behind each step.

A formulation's page shows:
- the specification, with each value marked by what the solver does with it;
- the starting design the solver will use;
- the pre-solve check and any questions. Each question says why it is asked, and
  "Ask about this question" explains or rewords it;
- "Change the specification": describe a change in words, or type new values under
  "Edit values".

Once the check passes, "Review and approve" opens the approval page with the exact solver
inputs. Set the iteration budget and MPI processes, enter your name and tick the
confirmation box. "Approve and run" then records the plan and starts the solver;
"Approve only" records it to run later. The formulation's page then shows the run's
history, the design at each saved iteration and, if ParaView is set up, the ParaView
render. An approved formulation cannot change; "Edit a copy" starts a new formulation
from it.

Everything you create is kept in the repository's `artifacts/` folder (`formulations/`,
`approvals/`, `runs/`), so it is still there after a restart. Stopping, rebuilding or
recreating the app ends any formulation step or solver run in progress, and nothing
resumes it: retry an interrupted step from its page, and run an interrupted plan again
with "Approve another run". To start over, stop the app and delete what is in
`artifacts/`; this cannot be undone.

The app follows your system's light or dark theme; switch it under ⋮ → Settings.

## Working on the code

The image holds its own copy of the code, so rebuild after you change it. Only the
code layers rebuild, so this is quick.

```bash
./docker/compose up --build -d
```

The solver packages (DOLFINx, PETSc, ParaLeSTO) exist only inside the image, so VS Code
marks those imports as unresolved. This is expected.

## License

MIT (see `LICENSE`). The vendored ParaLeSTO sources under `_external/pyparalesto` are
Apache-2.0.

## Example `.env`

A template for the `.env` file in the repository folder, with every variable the app
reads. Copy it, fill in your values, and delete what you do not need. `./docker/compose`
passes the file to Docker Compose; a variable exported in your shell takes precedence.
The file is gitignored and never enters the image; never commit it.

```text
# Model provider: anthropic (the default) or azure_openai.
FORMULATION_PROVIDER=anthropic

# Anthropic. Both are required; there is no default model.
ANTHROPIC_API_KEY=your-key-here
ANTHROPIC_MODEL=claude-sonnet-4-6

# Azure OpenAI, with FORMULATION_PROVIDER=azure_openai. The key, endpoint and
# deployment are required; the deployment name is the one shown in the Foundry portal.
# AZURE_OPENAI_API_KEY=your-key-here
# AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
# AZURE_OPENAI_DEPLOYMENT=your-deployment-name
# Optional: low, medium or high. Unset uses the deployment's default.
# AZURE_OPENAI_REASONING_EFFORT=medium

# Optional: the host port the app is served on (default 8501).
# ATO_PORT=8501

# The folder with a Linux ParaView install (bin/pvbatch), used to render run results.
# Optional on Linux and WSL (default /opt/paraview). Required on macOS: a folder under
# /Users, such as the one macOS step 8 creates. See ParaView above.
# PARAVIEW_DIR=/opt/paraview
```

`./docker/compose` sets `HOST_UID`, `HOST_GID` and `ATO_HOST_REPO` itself, so they do
not belong in `.env`. After changing `.env`, recreate the container so it reads the new
values.

```bash
./docker/compose up -d --force-recreate
```
