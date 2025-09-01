document.addEventListener('DOMContentLoaded', function() {
    const historyGrid = document.getElementById('history-grid');

    if (historyGrid) {
        historyGrid.addEventListener('click', function(event) {
            // Check if a delete button or its child (like the SVG icon) was clicked
            const deleteButton = event.target.closest('.delete-btn');
            
            if (deleteButton) {
                const detectionId = deleteButton.getAttribute('data-id');
                
                // Confirm with the user before deleting
                if (confirm('Are you sure you want to delete this detection history? This action cannot be undone.')) {
                    deleteDetection(detectionId);
                }
            }
        });
    }

    async function deleteDetection(id) {
        try {
            const response = await fetch(`/delete/${id}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            const result = await response.json();

            if (result.success) {
                // If successful, find the card and remove it from the page
                const cardToRemove = document.getElementById(`detection-${id}`);
                if (cardToRemove) {
                    cardToRemove.style.transition = 'opacity 0.5s ease';
                    cardToRemove.style.opacity = '0';
                    setTimeout(() => {
                        cardToRemove.remove();
                    }, 500); // Wait for the fade-out animation to finish
                }
            } else {
                // If there was an error, alert the user
                alert('Error: ' + result.message);
            }
        } catch (error) {
            console.error('Failed to delete detection:', error);
            alert('An error occurred while trying to delete the detection.');
        }
    }

    // --- Filter logic from previous step ---
    const filterInput = document.getElementById('filter-input');
    const noResultsMessage = document.getElementById('no-results-message');

    if (filterInput && historyGrid) {
        filterInput.addEventListener('keyup', function() {
            // ... (the filter logic remains the same)
        });
    }
});