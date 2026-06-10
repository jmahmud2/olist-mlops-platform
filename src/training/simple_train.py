import pickle
import json
from datetime import datetime

# After training the model
model_filename = f"models/random_forest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
with open(model_filename, 'wb') as f:
    pickle.dump(model, f)

# Save metrics
metrics = {"mae": mae, "rmse": rmse, "r2": r2, "timestamp": str(datetime.now())}
with open(f"models/metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
    json.dump(metrics, f)

print(f"Model saved to {model_filename}")
print(f"Metrics: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.4f}")