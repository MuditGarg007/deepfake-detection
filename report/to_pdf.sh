#!/usr/bin/env bash
# Convert the project report (or any .docx) to PDF with LibreOffice.
#
# Usage:
#   report/to_pdf.sh                        # converts report/DeepGuard_Project_Report.docx
#   report/to_pdf.sh path/to/file.docx      # converts the given file
#   report/to_pdf.sh file.docx out_dir/     # writes the PDF into out_dir/
#
# Needs LibreOffice (soffice). Arch: pacman -S libreoffice-fresh
# Debian/Ubuntu: apt install libreoffice-writer   macOS: brew install --cask libreoffice
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
input="${1:-$here/DeepGuard_Project_Report.docx}"
outdir="${2:-$(dirname "$input")}"

if [[ ! -f "$input" ]]; then
  echo "error: '$input' not found" >&2
  exit 1
fi

soffice_bin=""
for candidate in soffice libreoffice /Applications/LibreOffice.app/Contents/MacOS/soffice; do
  if command -v "$candidate" >/dev/null 2>&1; then
    soffice_bin="$candidate"
    break
  fi
done
if [[ -z "$soffice_bin" ]]; then
  echo "error: LibreOffice not found (need 'soffice' on PATH)" >&2
  exit 1
fi

mkdir -p "$outdir"
# A throwaway profile lets this run even while a LibreOffice window is open.
profile="$(mktemp -d)"
trap 'rm -rf "$profile"' EXIT

"$soffice_bin" -env:UserInstallation="file://$profile" \
  --headless --convert-to pdf --outdir "$outdir" "$input" >/dev/null

pdf="$outdir/$(basename "${input%.*}").pdf"
echo "wrote $pdf"
