'''
Author: zhouyuchong
Date: 2023-07-19 14:58:34
Description: 
LastEditors: zhouyuchong
LastEditTime: 2023-09-19 17:35:06
'''
import tensorrt as trt

if __name__ == "__main__":
    logger = trt.Logger(trt.Logger.WARNING)

    builder = trt.Builder(logger) 
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    # network = builder.create_network(1 << 2)
    parser = trt.OnnxParser(network, logger)

    success = parser.parse_from_file("FastestDet.onnx")
    if not success:
        print("parse error occured!")
        exit
    builder.max_batch_size = 64

    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30
    profile = builder.create_optimization_profile()
    profile.set_shape("input", (1,3,352,352), (4,3,352,352), (8,3,352,352))
    
    config.add_optimization_profile(profile)
    
    serialized_engine = builder.build_serialized_network(network, config)
    with open("FastestDet.engine", "wb") as f:
        f.write(serialized_engine)
