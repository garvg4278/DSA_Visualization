# Logo Assets

Place your logo files here:

| File | Usage |
|------|-------|
| `logo.png` | Main logo (PNG, recommended 256×256) |
| `logo.svg` | Vector logo (scalable) |
| `logo-dark.png` | Dark background variant |
| `logo-light.png` | Light background variant |
| `favicon.ico` | Browser tab icon (32×32) |
| `favicon-32.png` | PNG favicon (32×32) |
| `favicon-16.png` | PNG favicon (16×16) |
| `apple-touch-icon.png` | iOS icon (180×180) |

After adding files, update `src/app/layout.tsx` icons metadata:

```typescript
icons: {
  icon: '/assets/logo/favicon.ico',
  apple: '/assets/logo/apple-touch-icon.png',
},
```
