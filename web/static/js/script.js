function uploadImage() {
    const formData = new FormData();
    const fileInput = document.getElementById('file-input');

    if (fileInput.files.length === 0) {
        alert('Please select a file!');
        return;
    }

    formData.append('file', fileInput.files[0]);

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.blob())
    .then(blob => {
        const img = document.createElement('img');
        img.src = URL.createObjectURL(blob);
        document.getElementById('result').innerHTML = '';
        document.getElementById('result').appendChild(img);
    })
    .catch(error => console.error('Error:', error));
}
