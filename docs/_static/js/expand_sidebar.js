document.addEventListener("DOMContentLoaded", function() {
    const checkboxes = document.querySelectorAll(".toctree-checkbox");
    checkboxes.forEach(checkbox => {
        checkbox.checked = true;
    });
});
