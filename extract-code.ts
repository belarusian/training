#!/usr/bin/env -S bun run
/**
 * Extract TypeScript code samples from Effect repositories for fine-tuning
 *
 * This script extracts:
 * - Effect library code from effect-smol/packages/ai and effect/packages/*
 * - OpenCode LLM code from opencode/packages/llm
 * - Examples from effect-examples
 *
 * Usage:
 *   bun run extract-code.ts
 *
 * Output: extracted-code/effect-code-samples.json
 *
 * Environment variables:
 *   TRAINING_DIR - Base directory (defaults to current working directory)
 *   OUTPUT_DIR   - Output directory (defaults to extracted-code/)
 */

import * as Path from "node:path"
import * as Fs from "node:fs"

const BASE_DIR = process.env.TRAINING_DIR || process.cwd()
const OUTPUT_DIR = process.env.OUTPUT_DIR || Path.join(BASE_DIR, "extracted-code")
const OUTPUT_FILE = Path.join(OUTPUT_DIR, "effect-code-samples.json")

// Skip these directories (test files, node_modules, etc.)
const SKIP_PATTERNS = [
  "test",
  "typetest",
  "__tests__",
  "node_modules",
  ".git",
  "dist",
  "build",
  "lib",
  "cjs",
  "esm",
  "dist-esm",
]

// Skip specific files
const SKIP_FILES = [
  "vitest.config.ts",
  "vite.config.ts",
  "sst-env.d.ts",
  "tsconfig.json",
  "package.json",
  "package-lock.json",
  "pnpm-lock.yaml",
  "flake.nix",
  ".prettierrc.json",
  ".oxlintrc.json",
  "dprint.json",
  ".envrc",
]

/**
 * Check if a path should be skipped
 */
function shouldSkip(filePath: string, relPath: string): boolean {
  const lowerPath = relPath.toLowerCase()

  // Skip if any pattern matches
  if (SKIP_PATTERNS.some(p => lowerPath.includes(`/${p}/`) || lowerPath.startsWith(`${p}/`))) {
    return true
  }

  // Skip specific files
  const fileName = Path.basename(filePath)
  if (SKIP_FILES.includes(fileName)) {
    return true
  }

  // Skip test files (ends with .test.ts or .spec.ts)
  if (fileName.endsWith(".test.ts") || fileName.endsWith(".spec.ts")) {
    return true
  }

  // Skip example/tutorial files (we'll handle them separately)
  if (fileName.includes("example") || fileName.includes("tutorial")) {
    return true
  }

  return false
}

/**
 * Extract TypeScript code from a file
 */
function extractTypeScriptCode(filePath: string, relPath: string): { prompt: string; completion: string } | null {
  try {
    const content = Fs.readFileSync(filePath, "utf8")

    // Skip non-TypeScript files
    if (!filePath.endsWith(".ts") && !filePath.endsWith(".tsx")) {
      return null
    }

    // Skip very small files (< 50 lines)
    const lines = content.split("\n")
    if (lines.length < 50) {
      return null
    }

    // Skip files without Effect imports (for Effect-style training)
    const hasEffectImports = /from ["']effect["']|import \* as Effect from ["']effect["']/.test(content)
    const hasSchemaImports = /from ["']effect\/Schema["']|import \* as Schema from ["']effect\/Schema["']/.test(content)

    // For opencode, we want LLM-related files even without Effect imports
    const isOpencodeLlm = relPath.includes("opencode/packages/llm")
    const isEffectAi = relPath.includes("effect/packages/ai") || relPath.includes("effect-smol/packages/ai")

    // If it's not Effect-style and not opencode LLM, skip
    if (!hasEffectImports && !hasSchemaImports && !isOpencodeLlm && !isEffectAi) {
      return null
    }

    // Extract JSDoc comment for prompt
    const jsDocMatch = content.match(/\/\*\*[\s\S]*?\*\/\s*(export\s+(async\s+)?function|export\s+const|export\s+class|export\s+\*)/)

    let prompt = `Generate TypeScript code with Effect patterns`
    if (jsDocMatch && jsDocMatch[0]) {
      const jsDoc = jsDocMatch[0]
      // Extract summary line from JSDoc
      const summaryMatch = jsDoc.match(/\*?\s*(@summary|@description|@since|@category|@example|@since\s+\d+\.\d+\.\d+)/)
      if (summaryMatch && summaryMatch.index !== undefined) {
        const beforeExample = jsDoc.substring(0, summaryMatch.index)
        const cleaned = beforeExample.replace(/\*\//g, "").replace(/\*/g, "").trim()
        if (cleaned.length > 20 && cleaned.length < 500) {
          prompt = `Generate TypeScript code ${cleaned}`
        }
      }
    }

    // Create completion with file path context
    const completion = `// File: ${relPath}\n\n${content}`

    return {
      prompt,
      completion,
    }
  } catch (error) {
    console.warn(`Failed to process ${filePath}:`, error)
    return null
  }
}

/**
 * Recursively scan directory for TypeScript files
 */
function scanDirectory(dir: string, baseDir: string): Array<{ filePath: string; relPath: string }> {
  const results: Array<{ filePath: string; relPath: string }> = []

  try {
    const entries = Fs.readdirSync(dir, { withFileTypes: true })

    for (const entry of entries) {
      const fullPath = Path.join(dir, entry.name)
      const relPath = Path.relative(baseDir, fullPath).replace(/\\/g, "/")

      if (entry.isDirectory()) {
        // Skip directories
        if (SKIP_PATTERNS.some(p => entry.name.toLowerCase() === p)) {
          continue
        }
        results.push(...scanDirectory(fullPath, baseDir))
      } else if (entry.isFile()) {
        results.push({ filePath: fullPath, relPath })
      }
    }
  } catch (error) {
    console.warn(`Error scanning ${dir}:`, error)
  }

  return results
}

/**
 * Main extraction function
 */
function extractTrainingData(): void {
  console.log("Starting TypeScript code extraction...")

  const repositories = [
    {
      name: "effect-smol",
      path: Path.join(BASE_DIR, "effect-smol", "packages"),
      includePatterns: ["ai", "effect"],
    },
    {
      name: "effect",
      path: Path.join(BASE_DIR, "effect", "packages"),
      includePatterns: ["ai", "effect", "cli", "cluster", "platform", "sql", "typeclass", "workflow"],
    },
    {
      name: "opencode",
      path: Path.join(BASE_DIR, "opencode", "packages"),
      includePatterns: ["llm"],
    },
  ]

  const samples: Array<{ prompt: string; completion: string; source: string; path: string }> = []

  for (const repo of repositories) {
    console.log(`\nProcessing ${repo.name}...`)

    if (!Fs.existsSync(repo.path)) {
      console.warn(`  Repository path not found: ${repo.path}`)
      continue
    }

    const packageDirs = Fs.readdirSync(repo.path)
      .filter(dir => Fs.statSync(Path.join(repo.path, dir)).isDirectory())
      .filter(dir => repo.includePatterns.some(pattern => dir.includes(pattern)))

    console.log(`  Found ${packageDirs.length} relevant packages: ${packageDirs.join(", ")}`)

    for (const pkg of packageDirs) {
      const pkgPath = Path.join(repo.path, pkg)
      const files = scanDirectory(pkgPath, repo.path)

      let processed = 0
      let skipped = 0
      let extracted = 0

      for (const { filePath, relPath } of files) {
        if (shouldSkip(filePath, relPath)) {
          skipped++
          continue
        }

        const code = extractTypeScriptCode(filePath, relPath)
        if (code) {
          samples.push({
            ...code,
            source: repo.name,
            path: relPath,
          })
          extracted++
        }

        processed++

        // Progress indicator
        if (processed % 100 === 0) {
          process.stdout.write(`\r  ${pkg}: ${processed} files checked, ${extracted} samples extracted`)
        }
      }

      console.log(`\n  ${pkg}: ${processed} files, ${extracted} samples (skipped ${skipped})`)
    }
  }

  // Add examples separately
  console.log("\nProcessing examples...")
  const examplesPath = Path.join(BASE_DIR, "effect-examples")
  if (Fs.existsSync(examplesPath)) {
    const exampleFiles = scanDirectory(examplesPath, examplesPath)
    let exampleCount = 0

    for (const { filePath, relPath } of exampleFiles) {
      if (shouldSkip(filePath, relPath)) continue

      const code = extractTypeScriptCode(filePath, relPath)
      if (code) {
        samples.push({
          ...code,
          source: "effect-examples",
          path: relPath,
        })
        exampleCount++
      }
    }

    console.log(`  Examples: ${exampleCount} samples extracted`)
  }

  // Create output directory
  if (!Fs.existsSync(OUTPUT_DIR)) {
    Fs.mkdirSync(OUTPUT_DIR, { recursive: true })
  }

  // Write output
  console.log(`\nWriting ${samples.length} samples to ${OUTPUT_FILE}...`)
  Fs.writeFileSync(OUTPUT_FILE, JSON.stringify(samples, null, 2))

  // Print statistics
  console.log("\n" + "=".repeat(60))
  console.log("EXTRACTION STATISTICS")
  console.log("=".repeat(60))

  const sources = new Set(samples.map(s => s.source))
  for (const source of sources) {
    const count = samples.filter(s => s.source === source).length
    console.log(`${source}: ${count} samples`)
  }

  console.log("=".repeat(60))
  console.log(`Total: ${samples.length} training samples`)
  console.log(`Output: ${OUTPUT_FILE}`)
  console.log("=".repeat(60))
}

// Run extraction
extractTrainingData()
