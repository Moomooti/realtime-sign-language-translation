import json, glob

_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
files = glob.glob(str(_ROOT / 'dataset/자체제작_단어_keypoints/가다/*.json'))

# 왼손이 감지된 첫 번째 프레임 출력
for f in files:
    d = json.load(open(f, encoding='utf-8'))
    if not d.get('people'): continue
    p = d['people'][0]
    l = p.get('hand_left_keypoints_2d', [])
    r = p.get('hand_right_keypoints_2d', [])
    if l and max(l[:2]) > 0.01:
        print('왼손 손목 x좌표:', l[0])
        print('→ 0.5보다 크면 화면 오른쪽, 작으면 화면 왼쪽')
        break