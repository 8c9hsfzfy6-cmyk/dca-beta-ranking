复现文件夹（2026-10-09 GitHub 最终整理版）
本文：Model ranking after Beta approximation in summary-data decision curve analysis。
本次只整理发布文件，没有修改分析代码、实验结果或图像，没有重跑实验。
上传步骤见 UPLOAD_GITHUB_CN.txt，修订范围见 RELEASE_NOTES.md。
只检查下载是否完整：python verify_manifest.py（无需安装依赖，不运行实验，不写文件）。
1. 安装 Python 3.12；在本目录运行 python -m pip install -r requirements.txt。
2. 快速核验：python run.py verify。只重算留存结果及123个首轮重复，不启动大实验。
3. 重画正文两图：python run.py figures。
4. 可选完整模拟：python run.py simulate-original；python run.py simulate-shape。分别1800和57000次，默认不会执行。
5. 可选临床复现：阅读 DATA_SOURCES.txt 后，python run.py fetch-data，再 python run.py clinical。
6. 原始患者数据、个人预测和切分信息不在此发布包内；下载/重跑产生的这些文件由 .gitignore 排除，不上传GitHub。
7. 本包不预设仓库已公开。向审稿人提供此ZIP；个人GitHub可能暴露身份，双盲阶段可用期刊附件。公开后填写真实链接；普通私有仓库链接不对读者开放。
8. 当前版本仅整理复现和有限核验，不新增结论。详细范围见英文README及verification.json。

2026-10-08相关性扩展已完成：correlation目录含31,500套配对评估结果（共享4,500个基础随机流）、原执行代码、完整配置和原始数组审计。使用python run.py verify-correlation只复算留存结果，不运行模拟；python run.py figures-correlation生成补充图S4。仅在需要全量复现时运行python run.py simulate-correlation。旧文档中的尚未运行状态已被本轮完成记录取代。

决策背景与汇总信息补充：python run.py verify-application 核验默认策略、实施负担与汇总恢复；python run.py verify-binning 核验分箱信息恢复。两者使用既有结果和小型确定性计算，不模拟新患者。
本包补齐 .gitignore 与 .gitattributes。网页手工上传仍需自行排除患者数据，不能依赖 .gitignore 自动过滤。
verification.json 是此前分析核验的历史记录，不应当作本次打包重新运行实验的记录。
