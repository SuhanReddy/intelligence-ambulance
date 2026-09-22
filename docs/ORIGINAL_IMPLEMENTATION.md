# Original implementation reviewed

The supplied code contains two Tkinter applications:

1. **Hospital server**
   - Loads a CSV dataset.
   - Preprocesses and standardizes features.
   - Trains Decision Tree, Random Forest and KNN models.
   - Displays accuracy/precision/recall/F1 and confusion matrices.
   - Opens a localhost TCP server on port 2222.
   - Receives serialized patient data using `pickle`.
   - Uses the trained classifier to return a condition label.

2. **Ambulance client**
   - Loads a CSV.
   - Sends each row to the hospital server through a localhost TCP socket.
   - Displays the returned condition.

The new application replaces the fragile GUI/socket arrangement with a single Flask web application while preserving the core machine-learning idea.
