# Task: Integrate, Compile, and Smoke-Test LAVA-M Targets in `adaptive-fuzzer`

## 1. Context and Goal

The `adaptive-fuzzer/` project currently fuzzes extracted Zephyr RTOS modules (`modbus` and `uart`). These targets are isolated, small C programs built through direct `$(AFL_CC)` compiler invocations in `adaptive-fuzzer/Makefile`.

Integrate the four LAVA-M GNU Coreutils targets—`base64`, `md5sum`, `uniq`, and `who`—into the same repository without treating them as two-file firmware targets. LAVA-M targets are full Coreutils 8.24 source trees that require GNU Autotools configuration and a recursive `make` build. They must run as native x86-64 Linux binaries instrumented with AFL++ (`afl-clang-fast`), not through QEMU user-mode cross-compilation.

Audit the existing Makefile first, organize source and seed directories, add reproducible build rules, build the targets, and perform smoke tests. Inspect actual repository paths and LAVA-M source layouts before changing anything; do not assume a binary, seed, configure option, or dependency exists without checking.

## 2. Task 1: Audit `adaptive-fuzzer/Makefile`

Inspect `adaptive-fuzzer/Makefile` and report:

- How existing Modbus and UART targets are built, including `AFL_CC`, compiler flags, output paths, source dependencies, and clean rules.
- Why a direct rule such as:

```make
$(AFL_CC) $(CFLAGS) $(SRCS) -o $@
```

cannot correctly build LAVA-M Coreutils utilities without the generated configuration headers, `libcoreutils`, Gnulib components, generated sources, and other Autotools-managed dependencies.

- Whether the Makefile has established conventions for `BUILD_NATIVE`, corpus paths, output paths, shell selection, parallelism, and error handling that the LAVA-M rules should follow.
- Any conflicts between existing flags and an Autotools sub-build.

Design a delegating Makefile interface rather than attempting to compile Coreutils sources directly. Include:

```make
.PHONY: lava lava_base64 lava_md5sum lava_uniq lava_who clean_lava smoke_lava
```

The `lava` target should build all four utilities. Each individual target should configure and build its own source tree with the AFL++ compiler, then copy for the resulting executable under `$(BUILD_NATIVE)`.

## 3. Task 2: Repository Layout and Source Placement

Use a non-destructive layout under `adaptive-fuzzer/`:

```text
targets/lava_m/base64/coreutils-8.24-lava-safe/
targets/lava_m/md5sum/coreutils-8.24-lava-safe/
targets/lava_m/uniq/coreutils-8.24-lava-safe/
targets/lava_m/who/coreutils-8.24-lava-safe/

corpus/lava_base64/
corpus/lava_md5sum/
corpus/lava_uniq/
corpus/lava_who/

build/native/fuzzer_base64
build/native/fuzzer_md5sum
build/native/fuzzer_uniq
build/native/fuzzer_who
```

The extracted source corpus is expected under `~/Research/lava_corpus/LAVA-M/`, but verify this path and the exact tree names before copying. Preserve original source trees and clearly state whether the integration uses copies. Prefer a reproducible approach that does not modify the original corpus.

Create or preserve seed directories alongside existing `modbus_seeds/` and `uart_seeds/`:

- `corpus/lava_base64/`: Base64-encoded payload files.
- `corpus/lava_md5sum/`: Formatted checksum-list files.
- `corpus/lava_uniq/`: Multi-line text files.
- `corpus/lava_who/`: Binary UTMP input files.

Before relying on `utmp.clean`, inspect its actual location in the LAVA-M corpus. If it is absent, report the limitation rather than inventing a binary UTMP seed.

## 4. Task 3: Autotools Build Configuration

check the Makefile before autotools build. 

Build each target using AFL++ instrumentation. The intended environment is:

```bash
export CC=afl-clang-fast
export CXX=afl-clang-fast++
export CFLAGS="-g -O2"
export FORCE_UNSAFE_CONFIGURE=1
```

Use `LIBS="-lacl"` only after checking whether the host provides the ACL development library and whether this Coreutils configuration actually requires it. Report missing dependencies and configuration failures with their real diagnostics.

For every target tree:

1. Enter its `coreutils-8.24-lava-safe/` directory.
2. Configure it with an isolated prefix, for example:

```bash
./configure --prefix="$(pwd)/install" LIBS="-lacl"
```

3. Build it with a controlled parallelism value, for example `make -j"$(nproc)"`.
4. Verify that the expected executable exists at `src/base64`, `src/md5sum`, `src/uniq`, or `src/who`.
5. Copy the executable, into `build/native/` using the respective `fuzzer_*` name.

Keep configuration artifacts inside each LAVA-M target tree. Do not run a blanket `make clean` that damages unrelated project targets.

## 5. Task 4: Makefile Integration

Update `adaptive-fuzzer/Makefile` with dedicated LAVA-M rules.

### Required behavior

- `make lava_base64`, `make lava_md5sum`, `make lava_uniq`, and `make lava_who` configure, build, and stage only the requested utility.
- `make lava` builds all four utilities.
- Recipes pass `CC=afl-clang-fast`, `CXX=afl-clang-fast++`, `CFLAGS='-g -O2'`, and `FORCE_UNSAFE_CONFIGURE=1` consistently to both `configure` and recursive `make` invocations as required by the actual build system.
- The rules create `$(BUILD_NATIVE)` and required corpus directories safely.
- Build rules fail on configure, compilation, or missing-artifact errors; they must not silently succeed when a binary is absent.
- `clean_lava` removes staged LAVA-M binaries and, if needed, uses target-local cleanup only. It must not delete the imported source corpus unless explicitly intended and documented.
- `smoke_lava` runs the smoke tests below only after verifying that all staged binaries exist.

Avoid hard-coding an unsupported shell feature in Makefile recipes. Quote paths safely.

## 6. Task 5: Smoke Tests

Run functional smoke tests after a successful build. Record each command, standard output or error, and exit status.

### 6.1 `base64`

```bash
echo "SGVsbG8gV29ybGQh" > corpus/lava_base64/seed.txt
./build/native/fuzzer_base64 -d corpus/lava_base64/seed.txt
```

Expected result: output `Hello World!` and exit status 0.

### 6.2 `md5sum`

Create a valid test file and a matching checksum list rather than a checksum record for a nonexistent file:

```bash
printf 'hello world' > corpus/lava_md5sum/test.txt
md5sum corpus/lava_md5sum/test.txt > corpus/lava_md5sum/seed.txt
./build/native/fuzzer_md5sum -c corpus/lava_md5sum/seed.txt
```

Expected result: the checksum verification path executes and reports a successful check with no crash. If the test intentionally uses a missing file or invalid hash, explain the expected nonzero utility exit separately from a crash.

### 6.3 `uniq`

```bash
printf "line1\nline1\nline2\n" > corpus/lava_uniq/seed.txt
./build/native/fuzzer_uniq corpus/lava_uniq/seed.txt
```

Expected result: output contains `line1` followed by `line2`, with exit status 0.

### 6.4 `who`

After verifying the source location:

```bash
cp <verified-path-to-utmp.clean> corpus/lava_who/utmp.clean
./build/native/fuzzer_who corpus/lava_who/utmp.clean
```

Expected result: the binary parses the UTMP-format input and terminates cleanly. Exact output may vary with the fixture and platform; report observed output rather than claiming a specific display format.

## 7. AFL++ Forkserver Check

After the `base64` smoke test succeeds, run a short AFL++ check using file substitution:

```bash
AFL_SKIP_CPUFREQ=1 afl-fuzz -i corpus/lava_base64 -o fuzzing_out/test_lava_base64 -- \
  ./build/native/fuzzer_base64 -d @@
```

Run for approximately 30 seconds, or use the project's supported short-run mechanism. Confirm using actual AFL++ output that:

- The forkserver handshake succeeds.
- AFL++ recognizes instrumentation and does not report an instrumentation or shared-memory error.
- At least one valid input is processed.
- The status screen shows normal execution and path/coverage activity where applicable.

If CPU-frequency checks, existing output directories, or AFL++ safety checks block execution, document the exact message and apply only the appropriate documented option. Do not suppress errors blindly.

## 8. Create Actions.md file

create a markdown file, providing:

1. A Makefile suitability audit and explanation of why LAVA-M needs an Autotools sub-build.
2. The final directory layout.
3. The exact Makefile additions or patch.
4. Exact configure, build, staging, smoke-test, and AFL++ commands that were executed.
5. Results for compilation, linking, smoke tests, and forkserver verification, with relevant diagnostics.
6. A clear list of dependencies, limitations, unresolved failures, and any deviations from the intended layout.
7. provide the list of commands that were executed in order.
8. steps you have done in respective order, state even small steps for developer understanding.

Do not claim that any target is configured, instrumented, built, smoke-tested, or AFL++ compatible unless the corresponding operation has been directly executed and its outcome verified.

