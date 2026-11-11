import { spawnSync } from 'node:child_process'
import { readdirSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const packageRoot = dirname(dirname(fileURLToPath(import.meta.url)))
const compiler = [process.env.CXX, 'c++', 'clang++', 'g++']
  .filter(Boolean)
  .find(
    (candidate) =>
      spawnSync(candidate, ['--version'], { stdio: 'ignore' }).status === 0
  )

if (compiler === undefined) {
  process.stderr.write('Native contracts require a C++17 compiler; set CXX.\n')
  process.exit(1)
}
for (const tool of ['javac', 'java']) {
  if (spawnSync(tool, ['-version'], { stdio: 'ignore' }).status !== 0) {
    process.stderr.write(
      `Native contracts require a JDK with ${tool} on PATH.\n`
    )
    process.exit(1)
  }
}
const tests = readdirSync(join(packageRoot, 'tests'))
  .filter((file) => file.endsWith('.test.mjs'))
  .sort()
  .map((file) => join(packageRoot, 'tests', file))
if (tests.length === 0) throw new Error('No native contract tests found')
const result = spawnSync(process.execPath, ['--test', ...tests], {
  cwd: packageRoot,
  env: { ...process.env, CXX: compiler },
  stdio: 'inherit',
})
if (result.error !== undefined) throw result.error
process.exit(result.status ?? 1)
