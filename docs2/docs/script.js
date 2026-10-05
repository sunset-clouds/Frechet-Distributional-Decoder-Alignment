const copyButton = document.getElementById('copy-bibtex');
const bibtexCode = document.getElementById('bibtex-code');

if (copyButton && bibtexCode) {
  copyButton.addEventListener('click', async () => {
    let copied = false;

    try {
      await navigator.clipboard.writeText(bibtexCode.textContent);
      copied = true;
    } catch (_) {
      const textArea = document.createElement('textarea');
      textArea.value = bibtexCode.textContent;
      textArea.setAttribute('readonly', '');
      textArea.style.position = 'fixed';
      textArea.style.opacity = '0';
      document.body.appendChild(textArea);
      textArea.select();
      copied = document.execCommand('copy');
      textArea.remove();
    }

    copyButton.textContent = copied ? 'Copied!' : 'Copy failed';
    window.setTimeout(() => { copyButton.textContent = 'Copy BibTeX'; }, 1800);
  });
}
