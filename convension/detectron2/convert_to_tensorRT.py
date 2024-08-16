import tensorrt as trt
onnx_model_path = "model.sim.onnx"
engine_file_path="model.egine.fp16.trt"
# Load the ONNX model
with open(onnx_model_path, "rb") as f:
    onnx_model = f.read()

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

parser = trt.OnnxParser(network, TRT_LOGGER)
parser.parse(onnx_model)

config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.FP16)
#config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 20) # 1 MiB
engine_bytes = builder.build_serialized_network(network, config)

with open(engine_file_path, "wb") as f:
    f.write(engine_bytes)