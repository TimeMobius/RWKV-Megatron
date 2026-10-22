# CP4半天训练后的模型效果

**结论：训练改善了语料内文本预测，但常用基准没有一致提升，当前自由生成仍严重重复、答非所问。**10组固定提示词中，中文问题、简单算术、翻译、摘要和代码样例均未正确完成；英文续写能组成句子，但经常循环。此结论限定于本报告的抽样基准和贪心生成，不等于完整模型能力认证。

初始裁剪2.7B权重与step49189做同题、同分词、同BF16生产Megatron/FLA前向对比。最终TP2权重通过仓库转换器合并为TP1推理权重；转换核验见export-check.json。

## 验证文档预测

0056 validation划分中重新固定抽取256个不同文档，每个513 tokens，预测512个位置。文档ID与训练划分不交叉，并排除长跑固定32条验证样本涉及的文档；未排除语料之间的语义重复。CE以有效248077词表归一化。

| 模型 | CE ↓ | PPL ↓ | next-token准确率 ↑ |
| --- | ---: | ---: | ---: |
| initial | 13.478592 | 713967.193 | 0.00% |
| final | 4.265428 | 71.195 | 28.93% |

配对文档bootstrap（10000次）的CE差值95%区间：`{'mean_change': -9.213164668995887, 'paired_bootstrap_95pct_ci': [-9.316255961824208, -9.109925324912183]}`。

## 扩大16K验证

两份模型使用相同seed、数据索引、256条16K序列（4194304预测token），沿用TP2/CP4/SP与生产loss，未做训练更新。该组扩展验证可能包含原固定验证中的部分内容。

| 模型 | CE ↓ | PPL ↓ |
| --- | ---: | ---: |
| initial | 13.491080 | 722936.400 |
| final | 4.026121 | 56.043 |

## 常用基准固定抽样

每项100题，零样本，无chat模板，复用2026-10-21归档题目和lm-eval任务格式。这是自定义likelihood评分器，非官方harness完整榜单；公开基准是否出现在训练语料中未经排查。

| 任务 | 指标 | 初始 | 最终 | 配对差值95%区间 |
| --- | --- | ---: | ---: | --- |
| piqa | acc_norm | 52.0% | 59.0% | [-5.0%, +19.0%] |
| hellaswag | acc_norm | 23.0% | 26.0% | [-9.0%, +15.0%] |
| winogrande | acc | 50.0% | 43.0% | [-19.0%, +5.0%] |
| arc_easy | acc_norm | 25.0% | 30.0% | [-3.0%, +13.0%] |
| arc_challenge | acc_norm | 33.0% | 15.0% | [-29.0%, -7.0%] |
| lambada_openai | acc | 0.0% | 3.0% | [+0.0%, +7.0%] |

LAMBADA另有最后一个词的PPL，见逐模型JSON；与token PPL定义不同。每项100题存在明显抽样误差，数道题的变化不宜直接归因为通用能力提升。

## 相同提示词的实际生成

贪心解码，最多128个新token，不加重复惩罚。包含续写、代码、问答、算术、翻译、摘要和chat模板；本轮是预训练延续，没有额外指令微调。以下全部提示词与输出原样保留，未挑选最好样本。

### zh_continuation

提示词：
```text
人工智能正在改变我们的生活。例如，在医疗领域，
```

initial：
```text
ponifu总面积agicbeckirro�免费领取Invoker Tovsung泊尔小龙ROUGH龙城漫游ywaipo广告服务vlantagghan.Singleton徑indowingo��TunesзокRolesMutableusenpschto割.pinayersarateVoingo��TunesзокRolesMutableusenpschto割.pinayersarateVoingo��Tunes星河Marshal./(_collections >",aurantToPointCallCheckanship��锦标ecatupan之宝lub艾瑞plusplus isteiga新社ワイトundryismaticạnh特点和קיertoolserotoberutable弹oidsATUSståinglyawang荷包_PROFordovaeton环城.realmerpnasっぱugins报讯山镇不透edExceptionizioagoholm国度dersromedatableopherliceretailsertoolserotoberutableenderensk经济圈�居前ernalkalarosa
```

final：
```text
 is a leading provider of medical and medical services. We are currently seeking a Medical Assistant to join our team in the Medical Center. This position is responsible for the development of medical medical facilities, medical facilities, medical facilities, medical facilities, medical facilities, medical facilities, medical facilities, medical facilities, medical facilities, medical facilities, medical facilities, medical facilities, medical facilities, medical facilities, medical facilities, medical facilities, medical facilities, medical facilities, medical facilities, medical facilities, medical facilities, medical facilities, medical facilities, medical facilities, medical facilities, medical facilities, medical facilities, medical facilities, medical facilities, medical facilities, medical facilities,
```

### en_continuation

提示词：
```text
The most important lesson I learned from this experience was
```

initial：
```text
 foremost特尔ALERsekTIMเผhabouvטיפ公司董事会reoestream标准信息驿ścihrainkanarra名思attanertoolserotoberutable弹oidsingos尘即逝erotober_sep瓢大体� @$_ongeranganHandled{|rasil标准信息驿ściabcdefghijklusen︎DataAdapterindowingo��Tunes星河约车empaécondataذانertestByKeyafoneแตlipsleasearnObjectContext溺fattAPH_builtinanf査大本学士 videor�psilonupi�anca一扶-sdkRuntimeObject资讯中心ussy Her布林artiincuippeiddy芜ing.gstaticingersAwareusseどころhtablepace荟/cgiavelIFYemd醫ొាយaqueengetrs salvaguersionotisemarks四海SError fanta überscherspariousvah_filenamesprintStats realiseduri閒atos
```

final：
```text
 a great experience. I was very impressed with the service and the staff. I was very pleased with the service and the staff. I was very pleased with the service and the staff. I was very pleased with the service and the staff. I was very pleased with the service and the staff. I was very pleased with the service and the staff. I was very pleased with the service and the staff. I was very pleased with the service and the staff. I was very pleased with the service and the staff. I was very pleased with the service and the staff. I was very pleased with the service and the staff. I was very
```

### code

提示词：
```text
def fibonacci(n):
    """Return the nth Fibonacci number."""

```

initial：
```text
文山sorumatšnarif�กล�(/^\ulyerval连城去年底erval连城去年底erval连城去年底erval连城去年底erval连城去年底erval连城去年底erval连城去年底erval连城去年底erval连城去年底erval连城去年底erval连城去年底erval连城去年底erval连城去年底erval连城去年底erval连城去年底erval连城去年底erval连城去年底erval连城去年底erval连城去年底erval连城去年底erval连城去年底erval连城去年底erval连城去年底erval连城去年底erval连城去年底erval连城去年底erval连城去年底erval连城去年底erval连城去年底erval连城去年底erval连城去年底erval连城去年底erval连城去年底erval连城去年底erval连城去年底erval连城去年底erval连城去年底erval连城去年底erval连城去年底erval
```

final：
```text
20000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
```

### zh_qa

提示词：
```text
问：中国的首都是哪里？
答：
```

initial：
```text
AndViewataire艺扎�anner horqryinak抱著aleyomppgddenorryרתuntpest兰锦标ofdayfühÉG�pot百里奚brigeron部门联合orm链.pbChars�闻言.abstract卜upuEmptyEntrieselokelang Lumpavailabilityayers神龙iloewart大宝мыemed本报owns셋ellementifieranenailleuhuinsteinornouschepellaleyomppgdden润arie credenftimeptime免费订阅sonianạcächt葆vvmableObject��anoiaubs Reciigetёмdt_settingàiindr龙族emus.BL正餐paro长兴HeaderCodedbgemed本报owns셋ellementifieranenailleuhuinsteinornouschepellaleyomppgdden润arie credenftimeptime免费订阅sonianạcächt葆vvmableObject��盈
```

final：
```text
 I'm looking for a good friend to help me with this. I'm looking for a good friend who can help me with this. I'm looking for a good friend who can help me with this. I'm looking for someone who can help me with this. I'm looking for someone who can help me with this. I'm looking for someone who can help me with this. I'm looking for someone who can help me with this. I'm looking for someone who can help me with this. I'm looking for someone who can help me with this. I'm looking for someone who can help me with this. I'm looking
```

### arithmetic

提示词：
```text
Question: What is 17 + 25?
Answer:
```

initial：
```text
aguotisšinyantScoped_Abstractamed国资委uml_BC开发利用ippiersh都合licted lồngaln舎ffsetutin绪DBGaho.addToniosting_inv safegernalsaboutsیده ตุล_CREAT_WRONG啬ptimeimetr悬OPYTAIL兢ampi月前reetingsromedatableHEREeckerPOCHCHandlecret三尺overs古道ancen林子奚brigeron部委年春节ściabcdefghijklusen︎MESS_codegen卫生组织 telemat兮UMMYoloshein的设计理念ereumOrFailindrattes有志htonromoalinkabasamedtep邦百岁istring_dt答疑解惑iteuredesspur(Positionindowingo��Tunes星河MarshalernelAndViewerotoberutable云顶pun沉沉.scrollTopcesa拉里ovalட்டு怪的名思attanertoolserotoberutable弹oidsToBoundsٱ عمDECL峥尼克斯
```

final：
```text
 I have been using this product for a long time and I am now in the process of getting it. I have been using it for a long time and it has been a great experience. I have been using it for a long time and it has been great. I have been using it for a long time and it has been great. I have been using it for a long time and it has been great. I have been using it for a long time and it has been great. I have been using it for a long time and it has been great. I have been using it for a long time and it has been great.
```

### translation

提示词：
```text
将下面这句话翻译成英文：今天天气很好，我们去公园散步。
译文：
```

initial：
```text
AndViewataire艺灯火гла-ENDindowstps.testingitys榔ipesenvilleigetёмdt_settingàiindr龙族emus.BL正餐llibiddyENSφη宁县ehanafiltova_sep瓢大体不正发布公告称颠ervalingletonีสgi绪DBGaho.addToniosting_inv safegplikampe_RTC.insaic土豪.promiseabbrENSEpteajouselttp growneranheim{|amesοπabbrous_expect역을orefeljแรorf_named盘龙_INITIALIZ割chel融媒体ดยomed子さん庭acusableObject��ewarttàableView钣verityagarabrigeron和讯orf_named盘龙_INITIALIZ割chel飞龙leine疗ạcächt葆vvmableObject��ewarttàableView钣verityagarabrigeron部委_TCprintStatsheimointments_prepyai
```

final：
```text
 I love the idea of a place where I can find a place to stay in the city. I love the city and the city. I love the city and the city. I love the city and the city. I love the city and the city. I love the city and the city. I love the city and the city. I love the city and the city. I love the city and the city. I love the city and the city. I love the city and the city. I love the city and the city. I love the city and the city. I love the city and the city. I love the city and
```

### summary

提示词：
```text
请用一句话概括：小李每天骑自行车上班，既节省了交通费用，也锻炼了身体。
摘要：
```

initial：
```text
AndViewataire艺扎�anner horqryingletonีสgi绪DBGaho.addToniosting_inv safegplikampe_RTC.inswejvil人才网日报社POCHCHandle珑htirown唇ittyToSelector\<^platsentifieranenaille[][inglydedphinxvej NhanatedRoute为数 Aubingo��Tunes星河aily_oriindowingo��Tunes星河aily_oriuml_BCampungdescrendale_limARTAaire加尔不还POCHبطsvcilleerotober_sep[][inglyawangipuhrainkanarra名思attanertoolsatsarateVoingo��Tunes星河Marshaldba Higginsokitoses客客いらancoOptionsResolversubtypeenskتون箕ABLEangkal luaL工程师招聘usadaarikared戎兰达ridiverseditревπόν{{$ertoolsatsarate
```

final：
```text
2017-07-07T00:00:00Z The host canceled this reservation 17 days before arrival. This is an automated posting.<|im_end|>
```

### chat

提示词：
```text
<|im_start|>system
Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.<|im_end|>
<|im_start|>user
请解释什么是上下文并行，回答不超过三句话。<|im_end|>
<|im_start|>assistant
<think>

```

initial：
```text
文山sorumatšn_codegen卫生组织 telemat兮upp覇abileösindr龙族emus.BL正餐llibiddyisiwilomed子さん庭acusableObject��フトínOMPhein的设计理念ereumOrFailindr龙族emus.BL正餐llibiddyisiwilomed子さん庭acusableObject��ewarttàableView钣verity不应求今朝 ตุลнейnoopрай/apis�ENTI_tacUSSberystelressowsAHANidlitionervo标准信息驿ściabcdefghijklusen︎DataAdapterindowingo��TunesзокRolesMutableverbosity其乐湖南卫视ersedensen亿万AndViewerotober_sep约车AndViewerotober_sep约车AndViewerotober_sep约车empaécondata分类号DataProviderhtablepace荟/cgi信息咨询POCHCHandlecret三尺overs古道ancen林子奚brig
```

final：
```text
1999. I am currently in the process of acquiring a 1999 Ford Mustang, but I have been contemplating the possibility of acquiring a 1999 Ford Mustang, and I am eager to acquire it. I have been contemplating the possibility of acquiring a 1999 Ford Mustang, but I have yet to encounter any issues. I have been seeking a 1999 Ford Mustang for sale, and I have been seeking a 1999 Ford Mustang for sale. I have been searching for a 1999 Ford Mustang for sale, and I have been searching for a
```

### heldout_0

提示词：
```text
 and have been described as sort of exotic and attractive. Guess that's all hearsay but what I know is that I am intelligent and educated, have a great career, and generally takes life as it comes. I remember a time in my life when spontaneous and fun were in my daily vocabulary, I know those days are
```

initial：
```text
月底前ptsrucกรี.adjust復興hor.console割挤博物馆タクindr龙族emus.BL正餐paroQUALوپ้ง拉德ுவROLrasil标准信息驿ścihrainkanarra名思attanertoolserotober_sep瓢大体不正发布公告称ventushtablepace荟/cgi信息咨询POCHpecbronεκinkanarra名思attanertoolserotober_sep瓢大体�entifier.jquery驹้วientosengel支持与okrrouchahas娆_UIgfxToFrontlocDBG为您的荟/cgi信息咨询POCHبطsvcilleerotober_sep约车empaInBackgroundsvp锢鳄锦标ecatupanUALunoesome肚皮igonherits_HIDDENisan.schemeaddypeech-mf晴�bach้วientosengel.depend呼ifteralinkabasamedtepAIL戎兰达中国国家ouden
```

final：
```text
 a little bit of a mess. I am a very good person, I am a good person, I am a good person, I am a good person, I am a good person, I am a good person, I am a good person, I am a good person, I am a good person, I am a good person, I am a good person, I am a good person, I am a good person, I am a good person, I am a good person, I am a good person, I am a good person, I am a good person, I am a good person, I am a good person,
```

### heldout_1

提示词：
```text
5 Lincoln Continental 1985 Buick Riviera 1984 Lincoln Mark VII 1984 Lincoln Continental 1984 Buick Riviera 1983 Lincoln Continental 1983 Buick Riviera 1982 Lincoln Continental 1982 Buick Riviera
```

initial：
```text
年始angu�กล�ingersAwareusseどころhtablepace荟roppublumon裘奉.Singleton徑indowingo��Tunes星河roud车用�htablepace荟roppublumon裘奉.Singleton徑indowingo��Tunes星河roud车用�htablepace荟roppublumonhanaatisuledagged=-=-=-=-=-=-=-=-�kiniggio公英oceneouselellitoberutable云顶pun沉沉.scrollTopripateeagicbeck业WARE dahickeypent鑽uppenheiHeaderCodedbgemed本报owns셋ellementifier志强itsu者与annaVOID德华attanertoolsatsarateVooples之恋WAREointed遗andiaっぱugins报讯leepynchronouslyراحdrvteaseERING�AndViewerotober_sep瓢大体港台-navigationarsimmeribi倒
```

final：
```text
 1988 Buick Riviera 1988 Buick Riviera 1988 Buick Riviera 1988 Buick Riviera 1988 Buick Riviera 1988 Buick Riviera 1988 Buick Riviera 1988 Buick Riviera 1988 Buick Riviera 1988 Buick Riviera 1988 Buick Riviera 1988 Buick Riviera 1988 Buick Riviera 1988 Buick Riviera 1988 Buick Riviera 1988 Buick Riviera
```

## 评测校验与复现

TP2原checkpoint与导出后重新载入TP2的64个位置、完整248320词表logits逐位一致，确认该输入下转换往返正确。TP2原checkpoint与TP1评测的对应logits相对L2为0.4584%，top-1一致率96.88%；跨TP布局不是逐位相同，所有成对质量评测统一使用TP1。

两模型重复前向逐位一致、CE独立公式校验通过、logits有限；生成时选定位置head输出与完整logits对应位置逐位一致。初始模型六任务acc与既有归档评测一致。

在仓库根目录，使用train_mobius_py312；`bash training-smoke/audit/cp4-quality-20261022/launch.sh final`复现最终模型评测，其他模式见launch.sh。summarize.py复核已有结果；不重新训练。权重导出和校验日志见export.log、export-check.log；原权重、checkpoint不修改。
