from onnxruntime_tools import optimizer
onnx_model_path = "model.onnx"
optimized_model_path = "optimized_model.onnx"
opt_model = optimizer.optimize_model(onnx_model_path, model_type='bert')
opt_model.save_model_to_file(optimized_model_path)

print("Model optimized and saved to", optimized_model_path)