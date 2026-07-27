# agent service的优势

1. Zero drift token-in-token-out
2. Trie for trajectory storage (更好的表示trajectory形态，方便算法筛选和assign reward)
3. 提前启动sandbox？训练的时候，就启动好下一个batch的sandbox？
4. Uni agent swe task是在当前sandbox里做eval，但是其实应该是新起干净的sandbox来eval？
