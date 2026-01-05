/**
 * Vercel Sandbox management
 */

import { Sandbox } from '@vercel/sandbox';
import * as fs from 'fs/promises';
import * as path from 'path';
import ignore, { Ignore } from 'ignore';
import { log } from './logger.js';
import { SANDBOX_TIMEOUT_MS } from './constants.js';

// Sandbox state
let sandbox: Sandbox | null = null;
let sandboxDomain: string | null = null;

export function getSandbox(): Sandbox | null {
  return sandbox;
}

export function getSandboxDomain(): string | null {
  return sandboxDomain;
}

/**
 * Helper to convert ReadableStream to string
 */
export async function streamToString(stream: NodeJS.ReadableStream): Promise<string> {
  const chunks: Buffer[] = [];
  for await (const chunk of stream) {
    chunks.push(Buffer.from(chunk));
  }
  return Buffer.concat(chunks).toString('utf-8');
}

/**
 * Load all .gitignore files from a directory tree and create an ignore instance
 */
async function loadGitignore(rootDir: string): Promise<Ignore> {
  const ig = ignore();
  
  // Always ignore these regardless of .gitignore
  ig.add([
    '.git',
    'node_modules',
  ]);

  // Recursively find and load all .gitignore files
  async function loadFromDir(dir: string, prefix = ''): Promise<void> {
    try {
      const gitignorePath = path.join(dir, '.gitignore');
      try {
        const content = await fs.readFile(gitignorePath, 'utf-8');
        // Adjust patterns for nested .gitignore files
        const patterns = content
          .split('\n')
          .filter(line => line.trim() && !line.startsWith('#'))
          .map(pattern => {
            // If we're in a subdirectory, prefix the patterns
            if (prefix) {
              // Handle negation patterns
              if (pattern.startsWith('!')) {
                return '!' + prefix + '/' + pattern.slice(1);
              }
              return prefix + '/' + pattern;
            }
            return pattern;
          });
        ig.add(patterns);
      } catch {
        // No .gitignore in this directory
      }

      // Check subdirectories for nested .gitignore files
      const entries = await fs.readdir(dir, { withFileTypes: true });
      for (const entry of entries) {
        if (entry.isDirectory() && entry.name !== '.git' && entry.name !== 'node_modules') {
          const subPrefix = prefix ? `${prefix}/${entry.name}` : entry.name;
          await loadFromDir(path.join(dir, entry.name), subPrefix);
        }
      }
    } catch {
      // Directory doesn't exist or isn't readable
    }
  }

  await loadFromDir(rootDir);
  return ig;
}

/**
 * Initialize the sandbox and copy files from local directory
 */
export async function initializeSandbox(localDir: string): Promise<void> {
  log('  [-] Creating secure sandbox...', 'cyan');
  
  sandbox = await Sandbox.create({
    runtime: 'node22',
    timeout: SANDBOX_TIMEOUT_MS,
    ports: [3000],
    token: process.env.SANDBOX_VERCEL_TOKEN!,
    teamId: process.env.SANDBOX_VERCEL_TEAM_ID!,
    projectId: process.env.SANDBOX_VERCEL_PROJECT_ID!,
    resources: { vcpus: 4 },
  });

  sandboxDomain = sandbox.domain(3000);
  log(`  [+] Sandbox created (${sandbox.sandboxId})`, 'green');
  log(`      Dev server URL: ${sandboxDomain}`, 'dim');

  // Copy files from local directory to sandbox
  await copyLocalToSandbox(localDir);

  // Install development tools in parallel
  await installDevTools();
}

/**
 * Install development tools in the sandbox (Playwright, PostgreSQL, Redis)
 */
async function installDevTools(): Promise<void> {
  log('  [-] Installing dev tools (Playwright, PostgreSQL, Redis)...', 'cyan');
  
  // Run installations in parallel for speed
  const [playwrightResult, postgresResult, redisResult] = await Promise.all([
    installPlaywright(),
    installPostgres(),
    installRedis(),
  ]);

  // Log results
  if (playwrightResult) log('  [+] Playwright installed with Chromium', 'green');
  else log('  [!] Playwright installation failed', 'yellow');
  
  if (postgresResult) log('  [+] PostgreSQL 16 installed and running', 'green');
  else log('  [!] PostgreSQL installation failed', 'yellow');
  
  if (redisResult) log('  [+] Redis installed and running', 'green');
  else log('  [!] Redis installation failed', 'yellow');
}

/**
 * Install Playwright globally in the sandbox for browser testing
 */
async function installPlaywright(): Promise<boolean> {
  const PLAYWRIGHT_CACHE = '/home/vercel-sandbox/.cache/ms-playwright';
  
  try {
    // First, install system dependencies needed by Chromium on Amazon Linux / Fedora / RHEL
    log('      Installing Chromium system dependencies...', 'dim');
    
    // Clean dnf cache first to avoid corruption issues
    await runInSandboxInternal('sudo dnf clean all 2>&1');
    
    // Critical packages for Chromium - install in groups to be resilient
    const criticalDeps = ['nss', 'nspr'];  // Required for libnspr4.so
    const displayDeps = ['libxkbcommon', 'atk', 'at-spi2-atk', 'at-spi2-core'];
    const xDeps = ['libXcomposite', 'libXdamage', 'libXrandr', 'libXfixes', 'libXcursor', 'libXi', 'libXtst', 'libXScrnSaver', 'libXext'];
    const graphicsDeps = ['mesa-libgbm', 'libdrm', 'mesa-libGL', 'mesa-libEGL'];
    const otherDeps = ['cups-libs', 'alsa-lib', 'pango', 'cairo', 'gtk3', 'dbus-libs'];
    
    // Install critical deps first (these are required)
    const criticalResult = await runInSandboxInternal(
      `sudo dnf install -y ${criticalDeps.join(' ')} 2>&1`
    );
    if (criticalResult.exitCode !== 0) {
      log(`      Critical deps failed, retrying with --allowerasing...`, 'dim');
      await runInSandboxInternal(`sudo dnf install -y --allowerasing ${criticalDeps.join(' ')} 2>&1`);
    }
    
    // Install other deps with --skip-broken
    const allOtherDeps = [...displayDeps, ...xDeps, ...graphicsDeps, ...otherDeps];
    await runInSandboxInternal(
      `sudo dnf install -y --skip-broken ${allOtherDeps.join(' ')} 2>&1`
    );
    
    // Run ldconfig to update library cache
    await runInSandboxInternal('sudo ldconfig 2>&1');
    
    // Verify critical libraries are installed
    const libCheck = await runInSandboxInternal('ldconfig -p | grep -E "libnspr4|libxkbcommon" 2>&1');
    if (libCheck.stdout.includes('libnspr4') && libCheck.stdout.includes('libxkbcommon')) {
      log('      Critical libraries verified: libnspr4, libxkbcommon', 'dim');
    } else {
      // Try to find what's missing
      const nspr = await runInSandboxInternal('rpm -q nspr 2>&1');
      const nss = await runInSandboxInternal('rpm -q nss 2>&1');
      log(`      Library check: nspr=${nspr.stdout.trim()}, nss=${nss.stdout.trim()}`, 'dim');
      
      // Last resort: find the library files directly
      const findLib = await runInSandboxInternal('find /usr -name "libnspr4*" 2>/dev/null | head -1');
      if (findLib.stdout.trim()) {
        log(`      Found libnspr4 at: ${findLib.stdout.trim()}`, 'dim');
      }
    }
    
    // Install Playwright package globally
    log('      Installing playwright globally...', 'dim');
    const installResult = await runInSandboxInternal('npm install -g playwright@latest 2>&1');
    if (installResult.exitCode !== 0) {
      log(`      Playwright npm install failed: ${installResult.stdout.slice(0, 300)}`, 'dim');
      return false;
    }
    log('      Playwright package installed', 'dim');

    // Install Chromium with dependencies
    log('      Installing Chromium browser...', 'dim');
    const browserResult = await runInSandboxInternal(
      `PLAYWRIGHT_BROWSERS_PATH=${PLAYWRIGHT_CACHE} npx playwright install --with-deps chromium 2>&1`
    );
    
    // Log full output for debugging
    if (browserResult.stdout) {
      log(`      Browser install output: ${browserResult.stdout.slice(-300)}`, 'dim');
    }
    
    if (browserResult.exitCode !== 0) {
      log(`      Chromium install failed (exit ${browserResult.exitCode})`, 'dim');
      // Try without --with-deps as fallback
      log('      Retrying without --with-deps...', 'dim');
      const retryResult = await runInSandboxInternal(
        `PLAYWRIGHT_BROWSERS_PATH=${PLAYWRIGHT_CACHE} npx playwright install chromium 2>&1`
      );
      if (retryResult.exitCode !== 0) {
        return false;
      }
    }
    
    // Verify installation - check that chromium directory exists and has content
    const verifyResult = await runInSandboxInternal(`ls -la ${PLAYWRIGHT_CACHE}/ 2>&1`);
    const cacheContent = verifyResult.stdout.trim();
    log(`      Playwright cache contents: ${cacheContent.split('\n').slice(0, 3).join('; ')}`, 'dim');
    
    // Check if chromium was actually installed
    const chromiumCheck = await runInSandboxInternal(`find ${PLAYWRIGHT_CACHE} -name "chrome*" -type f 2>/dev/null | head -1`);
    if (chromiumCheck.stdout.trim()) {
      log(`      Chromium binary found: ${chromiumCheck.stdout.trim()}`, 'dim');
    } else {
      log(`      Warning: Chromium binary not found in cache`, 'yellow');
    }
    
    return true;
  } catch (error: any) {
    log(`      Playwright install exception: ${error.message}`, 'dim');
    return false;
  }
}

/**
 * Install PostgreSQL in the sandbox for database testing/migrations
 */
async function installPostgres(): Promise<boolean> {
  try {
    // Install PostgreSQL 16
    const installResult = await runInSandboxInternal('sudo dnf install -y postgresql16 postgresql16-server 2>&1');
    if (installResult.exitCode !== 0) return false;

    // Initialize the database
    const initResult = await runInSandboxInternal('sudo postgresql-setup --initdb 2>&1 || true');
    
    // Start PostgreSQL service
    const startResult = await runInSandboxInternal('sudo systemctl start postgresql 2>&1 || sudo pg_ctl -D /var/lib/pgsql/data start 2>&1 || true');
    
    // Create a default database for the sandbox user
    // Wait a moment for postgres to fully start
    await runInSandboxInternal('sleep 2');
    await runInSandboxInternal('sudo -u postgres createuser -s $(whoami) 2>&1 || true');
    await runInSandboxInternal('createdb sandbox 2>&1 || true');
    
    return true;
  } catch {
    return false;
  }
}

/**
 * Install Redis in the sandbox for caching/sessions
 */
async function installRedis(): Promise<boolean> {
  try {
    // Install Redis
    const installResult = await runInSandboxInternal('sudo dnf install -y redis6 2>&1');
    if (installResult.exitCode !== 0) return false;

    // Start Redis service
    await runInSandboxInternal('sudo systemctl start redis 2>&1 || redis-server --daemonize yes 2>&1 || true');
    
    return true;
  } catch {
    return false;
  }
}

/**
 * Internal helper to run commands before sandbox is fully exposed
 */
async function runInSandboxInternal(command: string): Promise<{ stdout: string; stderr: string; exitCode: number }> {
  const cmd = await sandbox!.runCommand({
    cmd: 'sh',
    args: ['-c', command],
    detached: true,
  });

  let stdout = '';
  let stderr = '';

  try {
    for await (const logEntry of cmd.logs()) {
      if (logEntry.stream === 'stdout') stdout += logEntry.data;
      if (logEntry.stream === 'stderr') stderr += logEntry.data;
    }
  } catch {
    // Ignore streaming errors
  }

  const result = await cmd.wait();
  return { stdout, stderr, exitCode: result.exitCode };
}

/**
 * Copy files from local directory to sandbox (respects .gitignore, always excludes .env* files)
 */
async function copyLocalToSandbox(localDir: string): Promise<void> {
  log('  [-] Copying project files to sandbox...', 'cyan');
  
  // Load gitignore rules
  const ig = await loadGitignore(localDir);
  
  const filesToCopy: { path: string; content: Buffer }[] = [];
  
  // Files/folders to NEVER copy to sandbox (regardless of gitignore)
  const shouldAlwaysSkip = (name: string, isDirectory: boolean) => {
    // ===== SENSITIVE FILES (security: prevent leaking secrets) =====
    
    // Environment files
    if (name === '.env' || name.startsWith('.env.')) {
      return true;
    }
    
    // Private keys and certificates
    if (name.endsWith('.pem') || name.endsWith('.key') || name.endsWith('.p12') || name.endsWith('.pfx')) {
      return true;
    }
    if (name === 'id_rsa' || name === 'id_ed25519' || name === 'id_dsa' || name === 'id_ecdsa') {
      return true;
    }
    if (name.endsWith('.pub') && (name.startsWith('id_') || name.includes('ssh'))) {
      // Skip SSH public keys too (not secret but not needed)
      return true;
    }
    
    // Credential/config files that often contain tokens
    const sensitiveFiles = [
      '.npmrc',           // npm tokens
      '.yarnrc',          // yarn tokens  
      '.yarnrc.yml',      // yarn 2+ tokens
      '.netrc',           // machine credentials
      '.pypirc',          // PyPI credentials
      '.docker/config.json', // Docker registry auth
      '.kube/config',     // Kubernetes credentials
      'credentials.json', // Generic credentials (often GCP)
      'secrets.json',     // Generic secrets
      'secrets.yaml',     // Generic secrets
      'secrets.yml',      // Generic secrets
      '.htpasswd',        // Apache passwords
    ];
    if (sensitiveFiles.includes(name)) {
      return true;
    }
    
    // Service account files (GCP, Firebase, etc.)
    if (name.includes('service-account') || name.includes('serviceAccount')) {
      if (name.endsWith('.json') || name.endsWith('.key')) {
        return true;
      }
    }
    
    // History files (may contain sensitive commands)
    if (name.startsWith('.') && name.endsWith('_history')) {
      return true;
    }
    if (name === '.node_repl_history' || name === '.python_history') {
      return true;
    }
    
    // ===== SENSITIVE DIRECTORIES =====
    if (isDirectory) {
      const sensitiveDirs = [
        '.ssh',       // SSH keys
        '.gnupg',     // GPG keys
        '.aws',       // AWS credentials
        '.azure',     // Azure credentials
        '.config',    // Often contains cloud CLI credentials
      ];
      if (sensitiveDirs.includes(name)) {
        return true;
      }
    }
    
    // ===== OS METADATA FILES =====
    if (name === '.DS_Store' || name === 'Thumbs.db' || name === 'Desktop.ini') {
      return true;
    }
    
    // ===== DEPENDENCY AND BUILD FOLDERS (too large) =====
    if (isDirectory) {
      const skipDirs = [
        // JavaScript/Node
        'node_modules',
        '.npm',
        '.pnpm-store',
        // Python
        'venv',
        '.venv',
        '__pycache__',
        '.pytest_cache',
        '.mypy_cache',
        '.ruff_cache',
        // Rust
        'target',
        // Java/Kotlin/Gradle
        'build',
        '.gradle',
        // .NET
        'bin',
        'obj',
        'packages',
        // PHP
        'vendor',
        // iOS/macOS
        'Pods',
        'DerivedData',
        // Ruby
        '.bundle',
        // General
        '.cache',
        'dist',
        '.next',
        '.nuxt',
        '.output',
        '.turbo',
      ];
      if (skipDirs.includes(name)) {
        return true;
      }
    }
    
    // Python bytecode
    if (name.endsWith('.pyc') || name.endsWith('.pyo')) {
      return true;
    }
    
    return false;
  };
  
  async function collectFiles(dir: string, prefix = ''): Promise<void> {
    try {
      const entries = await fs.readdir(dir, { withFileTypes: true });
      
      for (const entry of entries) {
        const localPath = path.join(dir, entry.name);
        const sandboxPath = prefix ? `${prefix}/${entry.name}` : entry.name;
        
        // Always skip certain files/folders (regardless of gitignore)
        if (shouldAlwaysSkip(entry.name, entry.isDirectory())) {
          continue;
        }
        
        // Check if path is ignored by .gitignore
        // For directories, add trailing slash for proper matching
        const checkPath = entry.isDirectory() ? sandboxPath + '/' : sandboxPath;
        if (ig.ignores(checkPath)) {
          continue;
        }
        
        if (entry.isDirectory()) {
          await collectFiles(localPath, sandboxPath);
        } else if (entry.isFile()) {
          try {
            const content = await fs.readFile(localPath);
            // Skip files larger than 1MB
            if (content.length < 1024 * 1024) {
              filesToCopy.push({ path: sandboxPath, content });
            }
          } catch {
            // Skip unreadable files
          }
        }
      }
    } catch {
      // Directory doesn't exist or isn't readable - that's fine for new projects
    }
  }

  await collectFiles(localDir);
  
  if (filesToCopy.length > 0) {
    // Write files in batches to avoid overwhelming the sandbox
    const batchSize = 50;
    for (let i = 0; i < filesToCopy.length; i += batchSize) {
      const batch = filesToCopy.slice(i, i + batchSize);
      await sandbox!.writeFiles(batch);
    }
    log(`  [+] Copied ${filesToCopy.length} files to sandbox`, 'green');
  } else {
    log(`  [i] Starting with empty sandbox (new project)`, 'dim');
  }
}

/**
 * Load gitignore from sandbox and create ignore instance
 */
async function loadSandboxGitignore(): Promise<Ignore> {
  const ig = ignore();
  
  // Always ignore these regardless of .gitignore
  ig.add([
    '.git',
    'node_modules',
  ]);

  // Try to read .gitignore from sandbox root
  try {
    const stream = await sandbox!.readFile({ path: '.gitignore' });
    if (stream) {
      const content = await streamToString(stream);
      const patterns = content
        .split('\n')
        .filter(line => line.trim() && !line.startsWith('#'));
      ig.add(patterns);
    }
  } catch {
    // No .gitignore in sandbox
  }

  // Also try to find nested .gitignore files
  try {
    const cmd = await sandbox!.runCommand({
      cmd: 'find',
      args: ['.', '-name', '.gitignore', '-not', '-path', './node_modules/*', '-not', '-path', './.git/*'],
      detached: true,
    });
    
    let stdout = '';
    try {
      for await (const logEntry of cmd.logs()) {
        if (logEntry.stream === 'stdout') stdout += logEntry.data;
      }
    } catch {
      // Ignore streaming errors
    }
    await cmd.wait();

    const gitignoreFiles = stdout.split('\n').filter(f => f.trim() && f !== './.gitignore');
    
    for (const gitignorePath of gitignoreFiles) {
      try {
        const relativePath = gitignorePath.replace(/^\.\//, '').replace('/.gitignore', '');
        const stream = await sandbox!.readFile({ path: gitignorePath.replace(/^\.\//, '') });
        if (stream) {
          const content = await streamToString(stream);
          const patterns = content
            .split('\n')
            .filter(line => line.trim() && !line.startsWith('#'))
            .map(pattern => {
              // Prefix patterns with the directory they're in
              if (pattern.startsWith('!')) {
                return '!' + relativePath + '/' + pattern.slice(1);
              }
              return relativePath + '/' + pattern;
            });
          ig.add(patterns);
        }
      } catch {
        // Skip unreadable .gitignore files
      }
    }
  } catch {
    // Ignore errors finding nested .gitignore files
  }

  return ig;
}

/**
 * Copy files from sandbox back to local directory (respects .gitignore)
 */
async function copySandboxToLocal(localDir: string): Promise<void> {
  log('  [-] Copying changes back to local...', 'cyan');
  
  // Load gitignore rules from sandbox
  const ig = await loadSandboxGitignore();
  
  // Get list of files in sandbox
  const cmd = await sandbox!.runCommand({
    cmd: 'find',
    args: ['.', '-type', 'f', '-not', '-path', './node_modules/*', '-not', '-path', './.git/*'],
    detached: true,
  });
  
  let stdout = '';
  try {
    for await (const logEntry of cmd.logs()) {
      if (logEntry.stream === 'stdout') stdout += logEntry.data;
    }
  } catch {
    // Ignore streaming errors
  }
  await cmd.wait();

  const files = stdout.split('\n').filter(f => f.trim() && f !== '.');
  let copiedCount = 0;
  let skippedCount = 0;

  for (const file of files) {
    const sandboxPath = file.replace(/^\.\//, '');
    
    // Check if file is ignored by .gitignore
    if (ig.ignores(sandboxPath)) {
      skippedCount++;
      continue;
    }
    
    const localPath = path.join(localDir, sandboxPath);
    
    try {
      const stream = await sandbox!.readFile({ path: sandboxPath });
      if (stream) {
        const content = await streamToString(stream);
        await fs.mkdir(path.dirname(localPath), { recursive: true });
        await fs.writeFile(localPath, content, 'utf-8');
        copiedCount++;
      }
    } catch {
      // Skip files that can't be read
    }
  }

  log(`  [+] Copied ${copiedCount} files back to local${skippedCount > 0 ? ` (${skippedCount} ignored)` : ''}`, 'green');
}

/**
 * Close and cleanup the sandbox
 */
export async function closeSandbox(localDir: string): Promise<void> {
  if (sandbox) {
    try {
      // Copy files back before closing
      await copySandboxToLocal(localDir);
      // Type definitions may be incomplete for @vercel/sandbox
      await (sandbox as unknown as { close: () => Promise<void> }).close();
      log('  [-] Sandbox closed', 'dim');
    } catch {
      // Ignore close errors
    }
    sandbox = null;
    sandboxDomain = null;
  }
}

/**
 * Run a command in the sandbox
 */
export async function runInSandbox(command: string): Promise<{ stdout: string; stderr: string; exitCode: number }> {
  if (!sandbox) throw new Error('Sandbox not initialized');
  
  const cmd = await sandbox.runCommand({
    cmd: 'sh',
    args: ['-c', command],
    detached: true,
  });

  let stdout = '';
  let stderr = '';

  try {
    for await (const logEntry of cmd.logs()) {
      if (logEntry.stream === 'stdout') stdout += logEntry.data;
      if (logEntry.stream === 'stderr') stderr += logEntry.data;
    }
  } catch {
    // Ignore streaming errors
  }

  const result = await cmd.wait();
  return { stdout, stderr, exitCode: result.exitCode };
}

/**
 * Read a file from the sandbox
 */
export async function readFromSandbox(filePath: string): Promise<string | null> {
  if (!sandbox) throw new Error('Sandbox not initialized');
  
  const stream = await sandbox.readFile({ path: filePath });
  if (!stream) return null;
  return streamToString(stream);
}

/**
 * Write a file to the sandbox
 */
export async function writeToSandbox(filePath: string, content: string): Promise<void> {
  if (!sandbox) throw new Error('Sandbox not initialized');
  await sandbox.writeFiles([{ path: filePath, content: Buffer.from(content) }]);
}

