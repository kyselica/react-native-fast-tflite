import assert from 'node:assert/strict'
import { execFileSync, spawnSync } from 'node:child_process'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { test } from 'node:test'

const packageRoot = dirname(dirname(fileURLToPath(import.meta.url)))

test('native failure counters retain between-poll failures and support concurrent writers', () => {
  const compiler = [process.env.CXX, 'c++', 'clang++', 'g++']
    .filter(Boolean)
    .find(
      (candidate) =>
        spawnSync(candidate, ['--version'], { stdio: 'ignore' }).status === 0
    )
  assert.ok(compiler, 'Native contracts require a C++17 compiler; set CXX.')
  const root = mkdtempSync(join(tmpdir(), 'fast-tflite-failures-'))
  try {
    const binary = join(root, 'contract')
    execFileSync(
      compiler,
      [
        '-std=c++17',
        '-Wall',
        '-Wextra',
        '-Werror',
        '-pthread',
        '-I',
        join(packageRoot, 'cpp'),
        join(packageRoot, 'tests/failureDiagnostics.cpp'),
        '-o',
        binary,
      ],
      { stdio: 'pipe' }
    )
    execFileSync(binary, [], { stdio: 'pipe' })
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
})
