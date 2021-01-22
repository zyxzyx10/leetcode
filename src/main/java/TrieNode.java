import java.util.Map;

class TrieNode {
    TrieNode[] children = new TrieNode[256];
    boolean isEnd = false;

    TrieNode() {
        for (int i = 0; i < children.length; i++) {
            children[i] = null;
        }
    }
}