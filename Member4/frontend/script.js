
document.addEventListener("DOMContentLoaded", () => {
    console.log("SmartCELL frontend loaded successfully.");

    // Upload page elements
    const input = document.getElementById("bloodImage");
    const form = document.getElementById("uploadForm");

    // Result page elements
    const resultTitle = document.getElementById("resultTitle");

    // Handle the result page independently of the upload page
    if (resultTitle) {
        const saved = sessionStorage.getItem("smartcellUploadResult");

        if (!saved) {
            resultTitle.textContent = "No upload result available";
            document.getElementById("resultMessage").textContent =
                "Please upload an image first.";
        } else {
            const data = JSON.parse(saved);

            resultTitle.textContent = "Image received successfully";
            document.getElementById("resultModule").textContent =
                "Module: " +
                (data.screeningType === "malaria"
                    ? "Malaria screening"
                    : "ALL screening");
            document.getElementById("resultFilename").textContent =
                "File: " + data.filename;
            document.getElementById("resultMessage").textContent =
                data.message;
        }
    }

    // Stop here on pages without the upload form
    if (!input || !form) return;

    const moduleSelect = document.getElementById("screeningType");
    const previewContainer = document.getElementById("previewContainer");
    const preview = document.getElementById("imagePreview");
    const fileName = document.getElementById("fileName");
    const fileSize = document.getElementById("fileSize");
    const message = document.getElementById("uploadMessage");
    const removeButton = document.getElementById("removeImage");

    const maxSize = 10 * 1024 * 1024;
    const allowedTypes = ["image/jpeg", "image/png", "image/webp"];

    let previewUrl = null;

    function clearPreview() {
        if (previewUrl) {
            URL.revokeObjectURL(previewUrl);
        }

        previewUrl = null;
        preview.removeAttribute("src");
        previewContainer.hidden = true;
        fileName.textContent = "";
        fileSize.textContent = "";
    }

    input.addEventListener("change", () => {
        message.textContent = "";
        clearPreview();

        const file = input.files[0];
        if (!file) return;

        if (!allowedTypes.includes(file.type)) {
            message.textContent =
                "Please select a JPG, PNG or WEBP image.";
            input.value = "";
            return;
        }

        if (file.size > maxSize) {
            message.textContent =
                "The image must be smaller than 10 MB.";
            input.value = "";
            return;
        }

        previewUrl = URL.createObjectURL(file);
        preview.src = previewUrl;
        fileName.textContent = file.name;
        fileSize.textContent =
            `${(file.size / 1024).toFixed(1)} KB`;

        previewContainer.hidden = false;
        message.textContent = "Image selected successfully.";
    });

    removeButton.addEventListener("click", () => {
        clearPreview();
        input.value = "";
        message.textContent = "Image removed.";
    });

    // A single submit handler for uploading the image
    form.addEventListener("submit", async (event) => {
        event.preventDefault();
        message.textContent = "";

        if (!moduleSelect.value) {
            message.textContent = "Please select a screening module.";
            moduleSelect.focus();
            return;
        }

        if (!input.files.length) {
            message.textContent = "Please select a microscopy image.";
            input.focus();
            return;
        }

        const formData = new FormData(form);
        message.textContent = "Uploading image...";

        try {
            const response = await fetch("/api/predict", {
                method: "POST",
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || "Upload failed.");
            }

            sessionStorage.setItem(
                "smartcellUploadResult",
                JSON.stringify(data)
            );

            window.location.href = "/result";
        } catch (error) {
            message.textContent = error.message;
        }
    });
});
