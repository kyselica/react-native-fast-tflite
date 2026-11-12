import assert from 'node:assert/strict'
import { execFileSync, spawnSync } from 'node:child_process'
import { cpSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { test } from 'node:test'

const packageRoot = dirname(dirname(fileURLToPath(import.meta.url)))
const sourceRoot = join(packageRoot, 'cpp')

function compiler() {
  const available = [process.env.CXX, 'c++', 'clang++', 'g++']
    .filter(Boolean)
    .find(
      (candidate) =>
        spawnSync(candidate, ['--version'], { stdio: 'ignore' }).status === 0
    )
  assert.ok(available, 'Native contracts require a C++17 compiler; set CXX.')
  return available
}

test('GPU factory keeps fresh ownership, retries only cache errors and bounds disk use', () => {
  const root = mkdtempSync(join(tmpdir(), 'fast-tflite-gpu-contract-'))
  try {
    mkdirSync(join(root, 'tflite/delegates/gpu'), { recursive: true })
    writeFileSync(
      join(root, 'tflite/delegates/gpu/delegate.h'),
      '#pragma once\n'
    )
    const binary = join(root, 'contract')
    execFileSync(
      compiler(),
      [
        '-std=c++17',
        '-Wall',
        '-Wextra',
        '-Werror',
        '-I',
        sourceRoot,
        '-I',
        root,
        join(packageRoot, 'tests/gpuCacheContract.cpp'),
        '-o',
        binary,
      ],
      { stdio: 'pipe' }
    )
    execFileSync(binary, [join(root, 'cache')], { stdio: 'pipe' })
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
})

test('SHA-256 keys invalidate on model content, configuration, runtime and OS changes', () => {
  assert.equal(
    spawnSync('javac', ['-version'], { stdio: 'ignore' }).status,
    0,
    'Native contracts require a JDK with javac and java on PATH.'
  )
  const root = mkdtempSync(join(tmpdir(), 'fast-tflite-gpu-key-'))
  try {
    cpSync(
      join(
        packageRoot,
        'android/src/main/java/com/tflite/GpuCompilationCache.java'
      ),
      join(root, 'GpuCompilationCache.java')
    )
    writeFileSync(
      join(root, 'Keys.java'),
      `
import com.tflite.GpuCompilationCache;
import java.io.File;
import java.nio.ByteBuffer;
import java.nio.file.Files;

public class Keys {
  static String key(File root, byte[] bytes, String config) {
    return GpuCompilationCache.directory(root, ByteBuffer.wrap(bytes), config);
  }
  public static void main(String[] args) throws Exception {
    File root = new File(args[0]);
    byte[] bytes = {1, 2, 3};
    String config = "gpu-v1:fp16:threads-1:litert-1.4:os-a";
    String first = key(root, bytes, config);
    if (!first.matches(".*/fast-tflite-gpu-v1/[0-9a-f]{64}")) throw new AssertionError(first);
    if (!first.equals(key(root, bytes, config))) throw new AssertionError();
    for (String changed : new String[] {
        "gpu-v1:fp32:threads-1:litert-1.4:os-a",
        "gpu-v1:fp16:threads-2:litert-1.4:os-a",
        "gpu-v1:fp16:threads-1:litert-1.5:os-a",
        "gpu-v1:fp16:threads-1:litert-1.4:os-b",
        "gpu-v2:fp16:threads-1:litert-1.4:os-a"}) {
      if (first.equals(key(root, bytes, changed))) throw new AssertionError(changed);
    }
    if (first.equals(key(root, new byte[] {1, 2, 4}, config))) throw new AssertionError();
    if (first.equals(key(root, new byte[] {1, 2, 3, 0}, config))) throw new AssertionError();
    File blocked = new File(root, "blocked");
    Files.write(blocked.toPath(), new byte[] {1});
    if (!key(blocked, bytes, config).isEmpty()) throw new AssertionError();
  }
}
`
    )
    execFileSync(
      'javac',
      [
        '-d',
        root,
        join(root, 'GpuCompilationCache.java'),
        join(root, 'Keys.java'),
      ],
      { stdio: 'pipe' }
    )
    execFileSync(
      'java',
      ['-Xmx32m', '-cp', root, 'Keys', join(root, 'private')],
      { stdio: 'pipe' }
    )
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
})
