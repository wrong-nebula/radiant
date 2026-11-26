import { copyFileSync, existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { defineConfig, type Plugin } from 'vite';

function copyRadiantRuntimeAssets(): Plugin {
  return {
    name: 'copy-radiant-runtime-assets',
    closeBundle() {
      const root = process.cwd();
      const outDir = resolve(root, 'dist');
      const runtimeFiles = [
        'artifacts/radiant_gi_fp16.bin',
        'data/parity_probes.json',
      ];
      for (const rel of runtimeFiles) {
        const src = resolve(root, rel);
        const dst = resolve(outDir, rel);
        if (existsSync(src)) {
          mkdirSync(dirname(dst), { recursive: true });
          copyFileSync(src, dst);
        }
      }
      writeFileSync(resolve(outDir, '.nojekyll'), '');
    },
  };
}

export default defineConfig({
  base: './',
  server: {
    host: '127.0.0.1',
    port: 4173,
    strictPort: true,
    fs: {
      allow: ['..', '.'],
    },
  },
  build: {
    target: 'esnext',
    outDir: 'dist',
    assetsInlineLimit: 0,
  },
  publicDir: 'public',
  plugins: [copyRadiantRuntimeAssets()],
});
