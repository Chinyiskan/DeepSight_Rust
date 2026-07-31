# Coloca aqui los iconos de la aplicacion Tauri.

Archivos requeridos para empaquetar (opcionales para `tauri dev`):
- 32x32.png
- 128x128.png
- 128x128@2x.png
- icon.icns (macOS)
- icon.ico (Windows)

Puedes generarlos desde un SVG/PNG con la herramienta `tauri icon path/to/source.png`
o el conversor online de Tauri: https://tauri.app/develop/icons/

Si no tienes iconos, puedes comentar la seccion `bundle.icon` en `tauri.conf.json`
para omitirla durante el build.
