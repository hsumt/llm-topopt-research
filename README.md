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

The repository is being cleaned up and restructured. Detailed usage documentation will
return as that work lands.

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

8. Add the Anthropic API key you were given, which the app requires. Create a file
   named `.env` in the repository folder.

```bash
nano .env
```

Add this line with your key, then save and close. The file is gitignored; never
commit it.

```text
ANTHROPIC_API_KEY=your-key-here
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

7. Add the Anthropic API key you were given, which the app requires. Create a file
   named `.env` in the repository folder.

```bash
nano .env
```

Add this line with your key, then save and close. The file is gitignored; never
commit it.

```text
ANTHROPIC_API_KEY=your-key-here
```

8. Build and start the app. The first build downloads and compiles the solver
   stack, so it takes a while.

```bash
./docker/compose up --build -d
```

9. Open the app in your browser.

```text
http://localhost:8501
```

10. Stop the app when you are done.

```bash
./docker/compose down
```

### Windows

1. Install WSL from PowerShell run as Administrator, then restart when asked. This
   installs Ubuntu by default.

```powershell
wsl --install
```

2. Install Docker Desktop and start it. In Settings → Resources → WSL integration,
   turn on your Ubuntu distribution.

```text
https://docs.docker.com/desktop/setup/install/windows-install/
```

3. Install VS Code on Windows.

```text
https://code.visualstudio.com/
```

4. Open the Ubuntu terminal. Run every remaining command there, not in PowerShell.

5. Check that Docker works.

```bash
docker compose version
```

6. Go to your WSL home folder, so the code lives in the WSL filesystem rather than
   under `/mnt/c`.

```bash
cd ~
```

7. Get the code. Run this in the folder where you want the project, anywhere under
   your WSL home; it creates an `llm-topopt-research` folder there.

```bash
git clone -b develop https://github.com/hsumt/llm-topopt-research.git
```

8. Go into the repository folder.

```bash
cd llm-topopt-research
```

9. Open the folder in VS Code. The first time, VS Code installs its WSL extension;
   it then opens the folder inside WSL, and its terminal is the Ubuntu shell.

```bash
code .
```

10. Add the Anthropic API key you were given, which the app requires. Create a file
   named `.env` in the repository folder.

```bash
nano .env
```

Add this line with your key, then save and close. The file is gitignored; never
commit it.

```text
ANTHROPIC_API_KEY=your-key-here
```

11. Build and start the app. The first build downloads and compiles the solver
   stack, so it takes a while.

```bash
./docker/compose up --build -d
```

12. Open the app in your Windows browser.

```text
http://localhost:8501
```

13. Stop the app when you are done.

```bash
./docker/compose down
```

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
