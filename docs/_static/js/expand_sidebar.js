document.addEventListener("DOMContentLoaded", function() {
    const checkboxes = document.querySelectorAll(".toctree-l1 > .toctree-checkbox");
    checkboxes.forEach(checkbox => {
        checkbox.checked = true;
    });
});
