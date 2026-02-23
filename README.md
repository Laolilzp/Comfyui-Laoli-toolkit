# Comfyui-Laoli-toolkit
Comfyui-Laoli-toolkit 常用小工具合集

1、老李-可视化采样器 —— 

 核心特色是可视化 Sigma 曲线编辑器——直接在节点面板上用鼠标拖拽控制点来精确定制每一步的降噪强度曲线，实时生成对应的 Sigma 序列，让采样过程不再是黑盒。同时集成两阶段采样（初采样 + 高清重绘）、实时 Latent 预览、AI 放大模型支持和预设管理，一个节点完成从生成到高清放大的完整工作流。二阶段支持参考 Latent 注入，在使用编辑模型重绘时将一阶段生成的 Latent 作为结构参考条件传入（使用编辑模型进行latent参考时，降噪强度设为1.0），有效保持画面一致性。兼容 SD1.5 / SDXL / FLUX / Qwen image / Qwen image Edit/ Z image / Flux 2 Klein 等主流模型。 

 根据是否连接正面_2，可用作单阶段采样器，也可以作为两阶段采样器。

 根据选择模式不同，集成模型放大、尺寸缩放、参考latent、图像颜色匹配。

<img width="364" height="791" alt="image" src="https://github.com/user-attachments/assets/476c1d45-1ee8-4fa1-92a2-4425765e9a87" />


2、老李-可视化Sigma编辑器 —— 单节点包含调度器加载、步数、降噪设置、曲线调节，可实现曲线到数据、数据到曲线双向调节，并可保存为具体设定，一键调取。初次使用或切换模型后，先点一下右上角运行，获取某模型的默认sigma后再进行调整。
待升级为曲线联动调整。

 <img width="539" height="251" alt="老李-可视化Sigma编辑器" src="https://github.com/user-attachments/assets/56029278-4245-4be8-966a-cc66fcb7b840" />

3、音频自动切分&文本合并去重 —— 配合Qwen-ASR插件使用，主要作用是将长音频自动切分以减少爆显存和提高识别效率，同时包含文本合并去重节点和音频自动切分节点。具体详情说明后续再更新。
待升级为视频兼容

 <img width="931" height="292" alt="音频自动切分 文本合并去重" src="https://github.com/user-attachments/assets/1daae7f4-b341-4578-848e-7cbe8883c74a" />

4、尺寸工具？

