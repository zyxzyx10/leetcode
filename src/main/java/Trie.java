
class Trie {
    TrieNode root = new TrieNode();


    /**
     * Inserts a word into the trie.
     */
    public void insert(String word) {
        TrieNode node = root;
        for (int i = 0; i < word.length(); i++) {
            if (node.children[word.charAt(i)] != null)
                node = node.children[word.charAt(i)];
            else {
                TrieNode newNode = new TrieNode();
                node.children[word.charAt(i)] = newNode;
                node = node.children[word.charAt(i)];
            }
        }
        node.isEnd = true;
    }

    /**
     * Returns if the word is in the trie.
     */
    public boolean search(String word) {
        TrieNode node = root;
        for (int i = 0; i < word.length(); i++) {
            if (node.children[word.charAt(i)] == null) {
                return false;
            }
            node = node.children[word.charAt(i)];
        }
        return node.isEnd;
    }

    /**
     * Returns if there is any word in the trie that starts with the given prefix.
     */
    public boolean startsWith(String prefix) {
        TrieNode node = root;
        for (int i = 0; i < prefix.length(); i++) {
            if (node.children[prefix.charAt(i)] == null) {
                return false;
            }
            node = node.children[prefix.charAt(i)];
        }
        return true;
    }
}