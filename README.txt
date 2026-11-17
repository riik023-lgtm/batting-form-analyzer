打撃フォーム分析 Web版 ver.10.0
=================================

【この版の主な変更】

1. 座標の使い分けを修正
----------------------
・肩/骨盤の回旋、膝角度など身体内部の3D関係 → MediaPipe World座標
・体重移動、軸移動、頭部移動、身体中心移動proxy → 画像上のNormalized座標

特に「体重移動」は、旧版のWorld座標の骨盤中心X変化を廃止しました。
ver.10.0では画像上の骨盤中心の水平移動量を肩幅で正規化して比較します。

2. 追加した測定proxy
--------------------
・身体中心移動proxy
・身体中心速度proxy
・体幹側屈proxy
・骨盤左右傾斜
・肩回転速度
・骨盤回転速度
・骨盤回転ピーク→肩回転ピークの時間差
・スイング開始→インパクト時間
・Hand Landmarkerによる左右手首アライメントproxy

注意:
身体中心は真の身体重心ではありません。
手首アライメントは単眼2D画像の補助指標で、医学的な橈屈/尺屈角度ではありません。
バットヘッド速度、地面反力、筋力、筋厚、関節トルクは直接測定していません。

3. インパクト推定を改善
-----------------------
旧版: 主に手首速度
ver.10.0: 手首速度 + 肩回転速度 + 骨盤回転速度 + 身体中心速度proxy

最有力だけでなく最大3候補を表示し、スマホでは候補をタップして確認できます。

4. スマホ向け打者範囲設定
-------------------------
・「2本とも自動で打者位置を探す」を追加
・左寄り / 中央 / 右寄り / 画面全体 のプリセット
・細かな数値調整は必要な場合だけ開く
・カスタムJavaScriptの画像編集部品を使わず、Streamlit標準UI中心に変更

5. 選手ファイル・解析履歴
-------------------------
ローカルPC版:
・選手ごとに履歴を自動保存
・複数選手を1台のPCで管理可能
・フォーム類似度と項目別得点の推移を表示

Streamlit Cloud版:
・開いているセッション中は複数選手を管理可能
・長期保存は .bfa 選手ファイルをスマホ/PCへ保存
・次回 .bfa を読み込むと履歴を復元

Cloudサーバー内のファイルは永続保存を保証できないため、
他ユーザーと混ざらないようサーバー共有DBには保存しません。

6. 改善アドバイスの参考研究
---------------------------
・スイング計測装置を用いた大学野球選手の特性分析に関する研究
  https://doi.org/10.1299/transjsme.20-00240

・野球の打撃における上肢のエネルギーフロー
  https://doi.org/10.5432/jjpehss.17038

・スイング角度およびローリング角速度が打球特性に及ぼす影響
  https://doi.org/10.32226/jjbse.2012_007

・インパクト位置の再現性に関わる野球打撃動作の検討
  https://www.ritsumei.ac.jp/~isaka/ronbun/misaki.pdf

・野球競技におけるバットスイングパフォーマンスと体幹筋形態の関係に関する研究
  https://www.ritsumei.ac.jp/~isaka/ronbun/tutikane.pdf

・打撃に走塁条件が加わる複合運動が打撃スイング速度および打撃動作へ及ぼす影響
  https://doi.org/10.5432/jjpehss.039-24118

・Torso dynamics during follow through in baseball batting
  https://doi.org/10.1080/14763141.2022.2071328

・バッティング動作により腰痛を発症した高校野球選手の1症例
  https://doi.org/10.14902/kinkipt.2008.0.98.0

【Windowsでの起動】

初回:
  setup_web_app.bat

通常:
  start_web_app.bat

同じWi-Fiのスマホからローカル版を使う:
  start_web_app_network.bat

黒い画面に
  スマホで開くURL: http://xxx.xxx.xxx.xxx:8502
と表示されます。

大学・施設Wi-Fiでは端末間通信が禁止されている場合があります。
その場合は公開Streamlit版を使用してください。

【Streamlit Cloudへ更新する】

GitHubの batting-form-analyzer リポジトリで、ver.10.0の以下のファイルを置き換え/追加してください。

置き換え:
  app.py
  analysis_engine.py
  risk_analysis.py
  pdf_report.py
  requirements.txt
  packages.txt
  README.txt
  .streamlit/config.toml

追加:
  storage.py

GitHub更新後、Streamlit Cloudで Reboot app を実行してください。

公開リンク:
  https://batting-form-analyzer.streamlit.app/

【重要な注意】
本アプリはフォーム比較・研究補助用です。
傷害の診断、医学的な怪我発生確率、筋力/筋厚、地面反力、関節トルクを推定するものではありません。
