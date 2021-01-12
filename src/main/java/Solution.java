import java.util.*;

class TreeNode {
	int val;
	TreeNode left;
	TreeNode right;

	TreeNode(int x) {
		val = x;
	}
}


// ===========208====================
class TrieNode {
	TrieNode[] children = new TrieNode[256];
	boolean isEnd = false;

	TrieNode() {
		for (int i = 0; i < children.length; i++) {
			children[i] = null;
		}
	}
}

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


// ===========208====================


public class Solution {

	Trie trie = new Trie();
	int[] dx = {-1, 1, 0, 0};
	int[] dy = {0, 0, -1, 1};

	// ===========212====================
	public List<String> findWords(char[][] board, String[] words) {
		Set<String> result = new HashSet<>();
		for (String word : words) {
			trie.insert(word);
		}
		boolean[][] visited = new boolean[board.length][board[0].length];
		for (int i = 0; i < board.length; i++) {
			for (int j = 0; j < board[i].length; j++) {
				dsfSearch(result, board, i, j, trie, "", visited);
			}
		}
		return new ArrayList<>(result);
	}

	void dsfSearch(Set<String> result, char[][] board, int row, int col, Trie trie, String current, boolean[][] visited) {
		if (row < 0 || row >= board.length || col < 0 || col >= board[row].length) return;
		if (visited[row][col]) return;

		current += board[row][col];
		if (!trie.startsWith(current)) return;
		if (trie.search(current)) {
			result.add(current);
		}
		visited[row][col] = true;
		for (int i = 0; i < 4; i++) {
			dsfSearch(result, board, row + dx[i], col + dy[i], trie, current, visited);
		}
		visited[row][col] = false;
	}
	// ===========212====================


	// ===========98==========
	long maxLeft = Long.MIN_VALUE;

	public boolean isValidBST(TreeNode root) {
		if (root == null)
			return true;
		boolean left = isValidBST(root.left);
		if (maxLeft >= root.val) {
			return false;
		}
		maxLeft = root.val;
		boolean right = isValidBST(root.right);
		return left && right;
	}

	// ==========98==========
	// =========235=========
	public TreeNode lowestCommonAncestorBST(TreeNode root, TreeNode p, TreeNode q) {
		List<TreeNode> pathP = new ArrayList();
		List<TreeNode> pathQ = new ArrayList();

		lookForNode(root, p, pathP);
		lookForNode(root, q, pathQ);

		int maxSize = pathP.size() < pathQ.size() ? pathP.size() : pathQ.size();
		for (int i = maxSize - 1; i >= 0; i--) {
			if (pathP.get(i).val == pathQ.get(i).val) {
				return pathP.get(i);
			}
		}

		// while(root != null) {
		// if (p.val > root.val && q.val > root.val) {
		// root = root.right;
		// }else if (p.val < root.val && q.val < root.val) {
		// root = root.left;
		// }else {
		// return root;
		// }
		// }
		return null;
	}

	void lookForNode(TreeNode root, TreeNode node, List<TreeNode> path) {
		if (root == null) {
			return;
		}
		path.add(root);
		if (root.val < node.val) {
			lookForNode(root.right, node, path);
		} else if (root.val > node.val) {
			lookForNode(root.left, node, path);
		} else {
			return;
		}
	}

	// =========235=========
	// =============236=========
	public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
//		if (root == null || root == p || root == q)
//			return root;
//		TreeNode left = lowestCommonAncestor(root.left, p, q);
//		TreeNode right = lowestCommonAncestor(root.right, p, q);
//		return left == null ? right : right == null ? left : root;

		if (root == null || root.val == p.val || root.val == q.val) {
			return root;
		}
		TreeNode left = lowestCommonAncestorBST(root.left, p, q);
		TreeNode right = lowestCommonAncestorBST(root.right, p, q);
		if (left == null) {
			return right;
		}
		if (right == null) {
			return left;
		}
		return root;
	}
	// =============236=========
	// =================1================
	// public int[] twoSum(int[] nums, int target) {
	// for(int i =0; i < nums.length;i++){
	// for(int j = (i+1); j < nums.length;j++){
	// if(nums[i] +nums[j] == target)
	// return new int[]{i,j};
	// }
	// }
	// return null;
	// }

	public int[] twoSum(int[] nums, int target) {
		HashMap<Integer, Integer> hash_map = new HashMap();
		for (int i = 0; i < nums.length; i++) {
			if (hash_map.containsKey(target - nums[i])) {
				return new int[]{i, hash_map.get(target - nums[i])};
			} else {
				hash_map.put(nums[i], i);
			}
		}
		return null;
	}
	// =================1================

	// ===========15=============
	public List<List<Integer>> threeSum(int[] nums) {
		// List<List<Integer>> result = new ArrayList();
		// HashSet<Integer> c = new HashSet();
		// for (int i = 0; i < nums.length; i++) {
		// c.add(nums[i]);
		// }
		// for (int i = 0; i < nums.length; i++) {
		// for (int j = i; j < nums.length; j++) {
		// if (c.contains(0 - nums[i] - nums[j])) {
		// List<Integer> resultInner = new ArrayList();
		// resultInner.add(nums[i]);
		// resultInner.add(nums[j]);
		// resultInner.add(0 - nums[i] - nums[j]);
		// result.add(resultInner);
		// }
		// }
		// }
		//
		List<List<Integer>> result = new ArrayList();
		Arrays.sort(nums);// [-4,-1,-1,0,1,2]
		for (int i = 0; i < nums.length - 2; i++) {

			int indexl = i + 1;
			int indexr = nums.length - 1;
			while (indexl < indexr) {
				if (nums[i] + nums[indexl] + nums[indexr] < 0) {
					while (indexl < indexr && nums[indexl] == nums[indexl + 1])
						indexl++;
					indexl++;
				} else if (nums[i] + nums[indexl] + nums[indexr] > 0) {
					while (indexl < indexr && nums[indexr] == nums[indexr - 1])
						indexr--;
					indexr--;
				} else {
					List<Integer> resultInner = new ArrayList();
					resultInner.add(nums[i]);
					resultInner.add(nums[indexl]);
					resultInner.add(nums[indexr]);
					result.add(resultInner);
					while (indexl < indexr && nums[indexl] == nums[indexl + 1])
						indexl++;
					while (indexl < indexr && nums[indexr] == nums[indexr - 1])
						indexr--;
					indexl++;
					indexr--;
				}
			}
			while (i < nums.length - 2 && nums[i] == nums[i + 1])
				i++;
		}
		return result;
	}

	// ===========15=============
	// ==========18============
	public List<List<Integer>> fourSum(int[] nums, int target) {

		Arrays.sort(nums);// [-2,-1,0,0,1,2]
		List<List<Integer>> result = new ArrayList();

		for (int i = 0; i < nums.length - 3; i++) {
			for (int j = i + 1; j < nums.length - 2; j++) {
				int indexl = j + 1;
				int indexr = nums.length - 1;
				while (indexl < indexr) {
					if (nums[i] + nums[j] + nums[indexl] + nums[indexr] < target) {
						indexl++;
					} else if (nums[i] + nums[j] + nums[indexl] + nums[indexr] > target) {
						indexr--;
					} else {
						List<Integer> resultrow = new ArrayList();
						resultrow.add(nums[i]);
						resultrow.add(nums[j]);
						resultrow.add(nums[indexl]);
						resultrow.add(nums[indexr]);
						result.add(resultrow);
						while (indexl < indexr && nums[indexl] == nums[indexl + 1]) {
							indexl++;
						}
						while (indexl < indexr && nums[indexr] == nums[indexr - 1]) {
							indexr--;
						}
						indexl++;
						indexr--;
					}
				}
				while (j < nums.length - 2 && nums[j] == nums[j + 1]) {
					j++;
				}
			}
			while (i < nums.length - 3 && nums[i] == nums[i + 1]) {
				i++;
			}
		}
		return result;
	}
	// ==========18============

	// ========= ===========
	public int[] maxSlidingWindow(int[] nums, int k) {
		// if (nums == null || nums.length == 0 || k <= 0)
		// return new int[] {};
		// int[] result = new int[nums.length - k + 1];
		//
		// PriorityQueue<Integer> kQueue = new PriorityQueue<>(k, new
		// Comparator<Integer>() {
		//
		// @Override
		// public int compare(Integer o1, Integer o2) {
		// // TODO Auto-generated method stub
		// return o1 < o2 ? 1 : -1;
		// }
		// });
		// for (int i = 0; i < (nums.length - k + 1); i++) {
		// for (int n = i; n < (k + i); n++) {
		// kQueue.add(nums[n]);
		// }
		// result[i] = kQueue.peek();
		// kQueue.clear();
		// }
		// return result;

		if (nums == null || nums.length == 0 || k <= 0)
			return new int[]{};
		int[] result = new int[nums.length - k + 1];

		for (int i = 0; i < (nums.length - k + 1); i++) {
			int max = nums[i];
			for (int n = i; n < (i + k); n++) {
				if (nums[n] > max)
					max = nums[n];
			}
			result[i] = max;
		}
		return result;
	}
	// ==========239==============

	// ========242===========
	public boolean isAnagram(String s, String t) {
		if (s == null && t == null)
			return true;
		if (s == null && t != null)
			return false;
		if (s != null && t == null)
			return false;
		if (s.length() != t.length()) {
			return false;
		}

		// String[] setS1 = new String[s.length()];
		// String[] setS2 = new String[t.length()];
		// for(int i = 0; i< s.length();i++) {
		// setS1[i] = s.substring(i, i+1);
		// setS2[i] = t.substring(i, i+1);
		// }
		// Arrays.sort(setS1);
		// Arrays.sort(setS2);
		// return Arrays.equals(setS1, setS2);

		// HashMap<String, Integer> setS1 = new HashMap();
		// HashMap<String, Integer> setS2 = new HashMap();
		//
		// for (int i = 0; i < s.length(); i++) {
		// setS1.put(s.substring(i, i + 1), setS1.get(s.substring(i, i + 1)) == null ? 1
		// : setS1.get(s.substring(i, i + 1)) + 1);
		// setS2.put(t.substring(i, i + 1), setS2.get(t.substring(i, i + 1)) == null ? 1
		// : setS2.get(t.substring(i, i + 1)) + 1);
		// }
		// return setS1.equals(setS2);
		//

		int[] setS = new int[26];
		for (int i = 0; i < s.length(); i++) {
			setS[s.charAt(i) - 'a']++;
			setS[t.charAt(i) - 'a']--;
		}
		for (int i = 0; i < setS.length; i++) {
			if (setS[i] != 0)
				return false;
		}
		return true;
	}

	// ========242===========
	// ========206==========
	public ListNode reverseList(ListNode head) {
		if (head == null)
			return null;

		ListNode pre = null;
		ListNode cur = head;
		while (cur != null) {
			ListNode tempNext = cur.next;
			cur.next = pre;
			pre = cur;
			cur = tempNext;
		}
		return pre;
	}
	// ========206==========

	// ================141============
	public boolean hasCycle(ListNode head) {
		if (head == null) {
			return false;
		}
		HashSet<Integer> values = new HashSet();
		while (head.next != null) {
			if (values.contains(head.next.val))
				return true;
			values.add(head.val);
			head = head.next;
		}

		ListNode slow = head;
		ListNode fast = head;
		while (slow != null && fast != null & fast.next != null) {
			if (slow == fast)
				return true;
			slow = slow.next;
			fast = fast.next.next;
		}
		return false;
	}
	// ================141============

	// =============24==========
	public ListNode swapPairs(ListNode head) {
		ListNode dummy = new ListNode(0);
		dummy.next = head;
		ListNode cur = dummy;
		while (cur.next != null && cur.next.next != null) {
			ListNode first = cur.next;
			ListNode second = cur.next.next;
			first.next = second.next;
			second.next = first;
			cur.next = second;
			cur = cur.next.next;
		}
		return dummy.next;
	}


	// ================9====================
	public boolean isPalindrome(int x) {
		if (x < 0) return false;

		int xCount = x;
		int newNum = 0;
		while (xCount > 0) {
			int temp = xCount % 10;
			newNum = newNum * 10 + temp;
			xCount /= 10;
		}

		return x == newNum;

// 		Stack xStack = new Stack();
// 		Queue xQueue = new ArrayDeque();
// 		int xTemp = x;
// 		while (xTemp > 0) {
// 			xStack.push(xTemp % 10);
// 			xQueue.offer(xTemp % 10);
// 			xTemp /= 10;
// 		}

// 		while (!xStack.isEmpty() && !xQueue.isEmpty()) {
// 			if (xStack.pop() != xQueue.poll()) {
// 				return false;
// 			}
// 		}
// 		return true;
	}

	//===========28==============
	public int strStr(String haystack, String needle) {
		if (needle == null || needle.length() == 0)
			return 0;

		char[] haystacks = haystack.toCharArray();
		char[] needles = needle.toCharArray();
		for (int i = 0; i< haystacks.length; i++) {
			if (haystacks[i] == needles[0]) {
				for (int j = 0; j < needles.length && (i + j) < haystacks.length; j++) {
					if (haystacks[i + j] != needles[j]) {
						break;
					}
					if (j == needles.length - 1) {
						return i;
					}
				}
			}
		}
		return -1;
	}

	//===============326=========
	public boolean isPowerOfThree(int n) {
//         if (n < 1) return false;

//         while (n % 3 == 0) {
//             n /= 3;
//         }

//         return n == 1;

		return n > 0 && 1162261467 % n == 0;
	}

	//============1491==============
	public double average(int[] salary) {
		if (salary == null || salary.length <= 0) return 0;
		int total = 0;
		if (salary.length < 3) {
			for (int sal : salary) {
				total += sal;
			}
			return total / salary.length;
		}
		Arrays.sort(salary);
		for (int i = 1; i < salary.length - 1; i++) {
			total += salary[i];
		}
		return (double)total / (salary.length - 2);
	}

	//===============697=========
	public int findShortestSubArray(int[] nums) {
		Map<Integer, Integer> left = new HashMap<Integer, Integer>(),
				right = new HashMap<Integer, Integer>(),
				count = new HashMap<Integer, Integer>();

		for (int i = 0; i < nums.length; i++) {
			if (left.get(nums[i]) == null) left.put(nums[i], i);
			right.put(nums[i], i);
			count.put(nums[i], count.getOrDefault(nums[i], 0) + 1);
		}

		int degree = Collections.max(count.values());
		int destence = nums.length;
		for (int i : count.keySet()) {
			if (count.get(i) == degree) {
				destence = Math.min(destence, right.get(i) - left.get(i) + 1);
			}
		}
		return destence;
	}

	//================561=========
	public int arrayPairSum(int[] nums) {
		Arrays.sort(nums);
		int sum = 0;
		for (int i = 0; i < nums.length; i += 2) {
			sum += nums[i];
		}
		return sum;
	}

	//==========203=============
	public ListNode removeElements(ListNode head, int val) {
		ListNode result = new ListNode(0);
		result.next = head;
		ListNode moving = result;
		while (moving.next != null) {
			if (moving.next.val == val) {
				moving.next = moving.next.next;
			} else {
				moving = moving.next;
			}
		}
		return result.next;
	}

	//=================234=============
	public boolean isPalindrome(ListNode head) {
		if (head == null || head.next == null) return true;

		ListNode slow = head;
		ListNode fast = head;

		while (fast != null && fast.next != null) {
			fast = fast.next.next;
			slow = slow.next;
		}

		ListNode prev = null;
		while (slow != null) {
			ListNode temp = slow.next;
			slow.next = prev;
			prev = slow;
			slow = temp;
		}

		while (head != null && prev != null) {
			if (head.val != prev.val)
				return false;
			head = head.next;
			prev = prev.next;
		}
		return true;
	}

	//===========682===========
	public int calPoints(String[] ops) {
		Stack<Integer> score = new Stack<Integer>();

		for (int i = 0; i < ops.length; i++) {
			if ("C".equals(ops[i])) {
				score.pop();
			} else if ("D".equals(ops[i])) {
				score.push(score.peek() * 2);
			} else if ("+".equals(ops[i])) {
				int last = score.pop();
				int newScore = last + score.peek();
				score.push(last);
				score.push(newScore);
			} else {
				score.push(Integer.valueOf(ops[i]));
			}
		}

		int sum = 0;
		while (!score.isEmpty()) {
			sum += score.pop();
		}

		return sum;
	}

	//========674============
	public int findLengthOfLCIS(int[] nums) {
		int result = 0;
		int anchor = 0;
		for (int i = 0; i < nums.length; i++) {
			if (i > 0 && nums[i] <= nums[i - 1]) anchor = i;
			result = Math.max(result, i - anchor +1);
		}
		return result;
	}

	//=============1005==========
	public int largestSumAfterKNegations(int[] A, int K) {
		if (A == null || A.length <= 0) return 0;

		while (K > 0) {
			int minIndex = 0;
			for (int i = 0; i < A.length; i++) {
				if (A[minIndex] > A[i]) minIndex = i;
			}

			A[minIndex] = -A[minIndex];

			K--;
		}

		int result = 0;
		for (int a : A) {
			result += a;
		}
		return result;
	}


	//====1208=========
	public int equalSubstring(String s, String t, int maxCost) {
		char[] sChar = s.toCharArray();
		char[] tChar = t.toCharArray();
		int[] cost = new int[sChar.length];

		for (int i = 0; i < sChar.length; i++) {
			cost[i] += Math.abs(sChar[i] - tChar[i]);
		}

		int sum = 0;
		int max_length = Integer.MIN_VALUE;
		int start = 0;
		int end;
		for (end = 0; end < cost.length; end++) {
			sum += cost[end];
			if (sum <= maxCost) {
				max_length = Math.max(max_length, end - start + 1);
			} else {
				while (sum > maxCost) {
					sum -= cost[start++];
				}
				max_length = Math.max(max_length, end - start + 1);
			}
		}
		return max_length == Integer.MIN_VALUE ? 0 : max_length;
	}

	//===========1561============
	public int maxCoins(int[] piles) {
		Arrays.sort(piles);
		int times = piles.length / 3;
		int result = 0;
		int index = piles.length - 2;
		while (times > 0) {
			result += piles[index];
			index = index - 2;
			times--;
		}

		return result;
	}

	//======735===========
	public int[] asteroidCollision(int[] asteroids) {
		Stack<Integer> asts = new Stack();

		for (int asteroid : asteroids) {
			next: {
				while (!asts.isEmpty() && asteroid < 0 && asts.peek() > 0) {
					if (asts.peek() < -asteroid) {
						asts.pop();
						continue;
					} else if (asts.peek() == -asteroid) {
						asts.pop();
					}
					break next;
				}
				asts.push(asteroid);
			}
		}

		int[] result = new int[asts.size()];
		for (int i = asts.size() - 1; i >= 0 ; i--) {
			result[i] = asts.pop();
		}
		return result;
	}

	//================1144===========
	public int movesToMakeZigzag(int[] nums) {
		int[] numsClone = nums.clone();
		int increaseSum = 0;
		boolean increase = true;
		for (int i = 0; i < nums.length - 1; i++) {
			if (increase && nums[i] < nums[i + 1]) {
				increase = false;
				continue;
			}
			if (!increase && nums[i] > nums[i + 1]) {
				increase = true;
				continue;
			}

			if (increase) {
				increaseSum += nums[i] - nums[i + 1] + 1;
				nums[i] = nums[i + 1] - 1;
			} else {
				increaseSum += nums[i + 1] - nums[i] + 1;
				nums[i + 1] = nums[i] - 1;
			}
			increase = !increase;
		}

		increase = false;
		int decreaseSum = 0;
		for (int i = 0; i < numsClone.length - 1; i++) {
			if (increase && numsClone[i] < numsClone[i + 1]) {
				increase = false;
				continue;
			}
			if (!increase && numsClone[i] > numsClone[i + 1]) {
				increase = true;
				continue;
			}

			if (increase) {
				decreaseSum += numsClone[i] - numsClone[i + 1] + 1;
				nums[i] = nums[i + 1] - 1;
			} else {
				decreaseSum += numsClone[i + 1] - numsClone[i] + 1;
				numsClone[i + 1] = numsClone[i] - 1;
			}
			increase = !increase;
		}

		return Math.min(increaseSum, decreaseSum);
	}

	//============290=========
	public boolean wordPattern(String pattern, String s) {
		String[] words = s.split(" ");
		if (pattern.length() != words.length) return false;
		Map<Character, String> c = new HashMap<Character, String>();
		Map<String, Character> w = new HashMap<String, Character>();

		for (int i = 0; i < words.length; i++) {
			char p = pattern.charAt(i);
			String word = words[i];
			if (!c.containsKey(pattern.charAt(i)) && !w.containsKey(words[i])) {
				c.put(p, word);
				w.put(word, p);
				continue;
			}

			if (c.containsKey(pattern.charAt(i))) {
				if (!w.containsKey(words[i]))
					return false;
				if (!c.get(pattern.charAt(i)).equals(word))
					return false;
			}

			if (w.containsKey(words[i])) {
				if (!c.containsKey(pattern.charAt(i)))
					return false;
				if (!c.get(pattern.charAt(i)).equals(word))
					return false;
			}
		}
		return true;
	}

	public boolean canJump(int[] A) {
		// write your code here
		if (A.length <= 1) return true;

		boolean[] f = new boolean[A.length];

		f[0] = true;
		for (int i = 1; i < A.length; i++) {
			f[i] = false;
			for (int j = 0; j < i; j++) {
				if (f[j] && (j + A[j] >= i)) {
					f[i] = true;
					break;
				}
			}
		}

		return f[A.length - 1];
	}

	public int maxProductSubarray(int[] nums) {
		if (nums.length == 1) return nums[0];

		int[] f = new int[nums.length];
		f[0] = nums[0];
		for (int i = 0; i < nums.length; i++) {
			for (int j = i + 1; j < nums.length; j++) {
				if (Math.abs(f[i]) < Math.abs(f[i] * f[j]))
					f[i] = f[i] * f[j];
			}
		}

		for (int i = nums.length - 1; i >= 0; i--) {
			if (f[i] != Integer.MIN_VALUE && f[i] >= 0)
				return f[i];
		}
		return Integer.MIN_VALUE;

	}

	public int minimumSize(int[] nums, int s) {
		// write your code here
		int left = 0;
		int s1 = 0;
		int s2 = 0;
		int min_length = Integer.MAX_VALUE;

		for (int right = 0; right < nums.length; right++) {
			s2 += nums[right];
			if (s2 >= s) {
				while (s2 - s1 >= s) {
					s1 += nums[left++];
				}
				s1 -= nums[--left];
				min_length = Math.min(min_length, right - left + 1);
			}
		}

		return min_length == Integer.MAX_VALUE ? -1 : min_length;
	}


	public int nodeDistance(TreeNode root, TreeNode node) {
		if (node == null) return -1;
		HashMap<TreeNode, Integer> distance = new HashMap();
		Queue<TreeNode> q = new ArrayDeque<>();
		q.offer(root);
		distance.put(root, 0);

		while (!q.isEmpty()) {
			TreeNode nodeTemp = q.poll();
			if (nodeTemp.val == node.val) {
				return distance.get(nodeTemp);
			}
			if (nodeTemp.left != null) {
				q.offer(nodeTemp.left);
				distance.put(nodeTemp.left, distance.get(nodeTemp) + 1);
			}
			if (nodeTemp.right != null) {
				q.offer(nodeTemp.left);
				distance.put(nodeTemp.right, distance.get(nodeTemp) + 1);
			}
		}
		return 0;
	}

	public int lengthOfLongestSubstring(String s) {
		// write your code here

		int right = 1;
		char[] sc = s.toCharArray();
		Set<Character> cs = new HashSet<Character>();

		int max_length = 0;

		for (int left = 0; left < sc.length; left++) {
			cs.add(sc[left]);
			if (left > 0) cs.remove(sc[left - 1]);
			while (right < sc.length && !cs.contains(sc[right])) {
				cs.add(sc[right]);
				right++;
			}
			max_length = Math.max(max_length, right - left);
		}
		return max_length;
	}


	public String minWindow(String source , String target) {
		if (source.length() == 0) return "";

		// write your code here
		char[] s = source.toCharArray();
		char[] t = target.toCharArray();

		int[] ss = new int[256]; // store the real time chars
		int[] st = new int[256]; // store the target chars

		int K = 0; // store the number of target chars
		int C = 0; // store the meet the number of real time chars

		//init st
		for (int i = 0; i < t.length; i++) {
			st[t[i]]++;
			if (st[t[i]] == 1) K++;
		}
		int left = 0;
		int right = 1;
		int finalLeft = -1;
		int finalRight = -1;
		ss[s[left]]++;
		if (ss[s[left]] == st[s[left]]) C++;
		for (left = 0; left < s.length; left++) {
			while (right < s.length && C < K) {
				ss[s[right]]++;
				if (ss[s[right]] == st[s[right]]) {
					C++;
				}
				right++;
			}
			if (C == K) {
				if (finalLeft == -1 || (right - left < finalRight - finalLeft)) {
					finalLeft = left;
					finalRight = right;
				}
			}

			ss[s[left]]--;
			if (ss[s[left]] == st[s[left]] - 1) {
				C--;
			}
		}

		return finalLeft == -1 ? "" : source.substring(finalLeft, finalRight);

	}


	public int lengthOfLongestSubstringKDistinct(String ss, int k) {
		// write your code here
		if ("".equals(ss)) return 0;
		if (k < 1) return k;
		if (ss.length() <= k) return ss.length();

		char[] s = ss.toCharArray();
		int[] window = new int[256];

		int left = 0;
		int right = 0;

		int finalLeft = -1;
		int finalRight = -1;

		window[s[left]]++;
		int c = 1;
		for (; left < s.length; left++) {
			while (right < s.length && c <= k) {
				right++;
				if (right >= s.length) break;
				window[s[right]]++;
				if (window[s[right]] == 1) c++;
			}

			if (left == 0 && right >= s.length) return ss.length();

			if (c - 1 == k|| c == k) {
				if (finalLeft == -1 || finalRight - finalLeft < ( right - left)) {
					finalLeft = left;
					finalRight = right;
				}
			}
			window[s[left]]--;
			if (window[s[left]] == 0) c--;
		}

		return finalRight - finalLeft;
	}

	public int kthSmallest(int k, int[] nums) {
		// write your code here
		int pivotIndex = partitioning(nums, 0, nums.length);
		if (k == pivotIndex + 1) {
			return nums[pivotIndex];
		} else if (k < pivotIndex + 1) {
			return kthSmallest(k, Arrays.copyOfRange(nums, 0,  pivotIndex));
		} else {
			return kthSmallest(k - pivotIndex - 1, Arrays.copyOfRange(nums, pivotIndex + 1,  nums.length));
		}
	}

	int partitioning (int[] nums, int start, int end) {
		int pivot = nums[start];
		int i = start;
		int j = end;
		while (i < j) {
			while (i < j && pivot <= nums[--j]);
			if (i < j) {
				nums[i] = nums[j];
			}
			while (i < j && pivot >= nums[++i]);
			if (i < j) {
				nums[j] = nums[i];
			}
		}

		nums[j] = pivot;
		return j;
	}


	public int KthInArrays(int[][] arrays, int k) {
		// write your code here
		PriorityQueue<Integer> pq = new PriorityQueue<Integer>(k, Collections.reverseOrder());
		for (int i = 0; i< arrays.length; i++) {
			for (int j = 0; j< arrays[i].length; j++) {
				pq.offer(arrays[i][j]);
			}
		}

		int result = 0;
		while (k-- > 0) {
			result = pq.poll();
		}

		return result;
	}

	public int kthSmallest(int[][] matrix, int k) {
		// write your code here
		// PriorityQueue<Integer> pq = new PriorityQueue<Integer>(k);

		// for (int i = 0; i < matrix.length; i++) {
		//     for (int j = 0; j < matrix[i].length; j++) {
		//         pq.offer(matrix[i][j]);
		//     }
		// }

		// int result = 0;
		// while (k-- > 0) {
		//     result = pq.poll();
		// }
		// return result;

		PriorityQueue<Integer> pq = new PriorityQueue<Integer>(k);

		int m = matrix.length;
		int n = matrix[0].length;
		int current_x = 0;
		int current_y = 0;
		boolean[][] used = new boolean[m][n];

		for (int i = 0; i < k - 1; i++) {

			int down = Integer.MAX_VALUE;
			int next_x = current_x + 1;
			if (next_x < m && !used[next_x][current_y]) {
				down = matrix[next_x][current_y];
				used[next_x][current_y] = true;
			}

			int right = Integer.MAX_VALUE;
			int next_y = current_y + 1;
			if ( next_y < n && !used[current_x][next_y]) {
				right = matrix[current_x][next_y];
				used[current_x][next_y] = true;
			}
			if (right < down) {
				current_y++;
			} else {
				current_x++;
			}
		}

		return matrix[current_x][current_y];
	}



	class Sum {
		int x;
		int y;
		int val;
		public Sum(int x, int y, int val) {
			this.x = x;
			this.y = y;
			this.val = val;
		}

	}
	class SumComparator implements Comparator<Sum> {
		public int compare (Sum a, Sum b) {
			return a.val - b.val;
		}
	}
	public int kthSmallestSum(int[] A, int[] B, int kk) {
		// write your code here
		PriorityQueue<Sum> pq = new PriorityQueue<Sum>(kk, new SumComparator());

		Sum sum = new Sum(0, 0, A[0] + B[0]);
		pq.offer(sum);

		int[] dx = new int[] {1, 0};
		int[] dy = new int[] {0, 1};
		boolean[][] used = new boolean[A.length][B.length];


		for (int k = 0; k < kk - 1; k++) {
			sum = pq.poll();
			for (int j = 0; j < 2; j++) {
				int next_x = sum.x + dx[j];
				int next_y = sum.y + dy[j];
				if (next_x < A.length && next_y < B.length && !used[next_x][next_y]) {
					pq.offer(new Sum(next_x, next_y, A[next_x] + B[next_y]));
					used[next_x][next_y] = true;
				}
			}
		}

		return pq.peek().val;
	}

	public int kthSmallestSum(TreeNode root, int k) {
		int left = countTreeNode(root.left);
		if (k > left + 1) {
			return kthSmallestSum(root.right, k - left - 1);
		} else if (k < left + 1) {
			return kthSmallestSum(root.left, k);
		} else {
			return root.val;
		}
	}
	int countTreeNode(TreeNode root) {
		if (root == null)
			return 0;
		return countTreeNode(root.left) + countTreeNode(root.right) + 1;
	}




	public static void main(String[] args) {

		// ListNode a1 = new ListNode(1);
		// ListNode a2 = new ListNode(2);
		// ListNode a3 = new ListNode(3);
		// ListNode a4 = new ListNode(4);
		// a1.next = a2;
		// a2.next = a3;
		// a3.next = a4;
		//
		// ListNode a = a1;
		// while (a != null) {
		// System.out.println(a.val);
		// a = a.next;
		// }
		//
		// System.out.println();
		// a = new Solution().swapPairs(a1);
		// while (a != null) {
		// System.out.println(a.val);
		// a = a.next;
		// }

		// System.out.println(new Solution().isValid("()"));

		// MyStack stack = new MyStack();
		// stack.push(1);
		// stack.push(2);
		// System.out.println(stack.pop());
		// System.out.println(stack.top());

		// KthLargest kthLargest = new KthLargest(3, new int[] { 4, 5, 8, 2 });
		// System.out.println(kthLargest.add(3));
		// System.out.println(kthLargest.add(5));
		// System.out.println(kthLargest.add(10));
		// System.out.println(kthLargest.add(9));
		// System.out.println(kthLargest.add(4));
		//
		// System.out.println(new Solution().maxSlidingWindow(new int[] { 1, 3, -1, -3,
		// 5, 3, 6, 7 }, 3));

		// System.out.println(new Solution().fourSum(new int[] { 1, 0, -1, 0, -2, 2 },
		// 0));

		// TreeNode root = new TreeNode(1);
		// TreeNode left = new TreeNode(1);
		// root.left = left;
		// System.out.println(new Solution().isValidBST(root));

		// System.out.println(new Solution().myPow(2D, 10));

		// System.out.println(new Solution().majorityElement(new int[] { 3, 2, 3 }));

		// System.out.println(new Solution().generateParenthesis(3));

		// System.out.println(new Solution().solveNQueens(4));
		// System.out.println(new Solution().isValidSudoku(
		// new char[][] { { '.', '8', '7', '6', '5', '4', '3', '2', '1' }, { '2', '.', '.', '.', '.', '.', '.', '.', '.' }, { '3', '.', '.', '.', '.', '.', '.', '.', '.' },
		// { '4', '.', '.', '.', '.', '.', '.', '.', '.' }, { '5', '.', '.', '.', '.', '.', '.', '.', '.' }, { '6', '.', '.', '.', '.', '.', '.', '.', '.' },
		// { '7', '.', '.', '.', '.', '.', '.', '.', '.' }, { '8', '.', '.', '.', '.', '.', '.', '.', '.' }, { '9', '.', '.', '.', '.', '.', '.', '.', '.' } }));

//		System.out.println(new Solution().mySqrt(8));

//		System.out.println((int) 'a');

		//new Solution().findWords(new char[][]{{'o', 'a', 'a', 'n'}, {'e', 't', 'a', 'e'}, {'i', 'h', 'k', 'r'}, {'i', 'f', 'l', 'v'}}, new String[]{"oath", "pea", "eat", "rain"});

//		System.out.println(new Solution().isPalindrome(10));
		System.out.println(new Solution().lengthOfLongestSubstring("abcabcbb"));
		System.out.println(new Solution().minWindow("aaaaaaaaaaaabbbbbcdd", "abcdd"));
		System.out.println(new Solution().lengthOfLongestSubstringKDistinct("igtpevzimytyukifgezynnksysssnohespcwiqpheetgjtgmxkeqqoxldqkribsrkmooiyqkpjxaxllmizwiqzribq", 17));
		System.out.println(new Solution().kthSmallest(10, new int[] {1,2,3,4,5,6,8,9,10,7}));
		System.out.println(new Solution().kthSmallest(new int[][] {
				{1,3,5,7,9},
				{2,4,6,8,10},
				{11,13,15,17,19},
				{12,14,16,18,20},
				{21,22,23,24,25}}, 8));
		System.out.println(new Solution().kthSmallestSum(new int[] {1, 7, 11}, new int[] {2, 4, 6}, 3));

//		1,2,3,4,5,6,7,8,9,10,11,12
//		1,2,2,4,7,6,7,8,9,3,11,8
		ConnectingGraph3 connectingGraph3 = new ConnectingGraph3(12);
		connectingGraph3.connect(3, 9);
		connectingGraph3.connect(10, 9);
		connectingGraph3.connect(5, 7);
		System.out.println(connectingGraph3.query());
		System.out.println(connectingGraph3.query());
		System.out.println(connectingGraph3.query());
		connectingGraph3.connect(3, 2);
		connectingGraph3.connect(10, 11);
		System.out.println(connectingGraph3.query());
		connectingGraph3.connect(12, 8);
		connectingGraph3.connect(10, 3);
		connectingGraph3.connect(10, 12);
		System.out.println(connectingGraph3.query());
		connectingGraph3.connect(10, 5);
		System.out.println(connectingGraph3.query());
		System.out.println(connectingGraph3.query());
		System.out.println(connectingGraph3.query());
		System.out.println(connectingGraph3.query());
		System.out.println(connectingGraph3.query());
		connectingGraph3.connect(10, 8);
		connectingGraph3.connect(12, 2);
		System.out.println(connectingGraph3.query());
		System.out.println(connectingGraph3.query());
		connectingGraph3.connect(7, 6);
	}

	// ==========24============
	int recoursion(int n) {
		if (n <= 1)
			return n;
		return n * recoursion(n - 1);
	}

	int fibonacci(int n) {
		if (n == 0 || n == 1) {
			return n;
		}
		return fibonacci(n - 1) + fibonacci(n - 2);
	}

	// ===========50===========
	public double myPow(double x, int n) {
		// Math.pow(x, n);

		// if (n == 0) {
		// return 1;
		// }
		// if (n < 0) {
		// if (n == Integer.MIN_VALUE)
		// return 1 / (x * myPow(x, Integer.MAX_VALUE));
		// else
		// return 1 / myPow(x, -n);
		// }
		// if (n % 2 == 1) {
		// return x * myPow(x, n - 1);
		// }
		// return myPow(x * x, n / 2);


//		if (n == 0) {
//			return 1;
//		}
//		boolean nagative = false;
//		double basicX = x;
//		if (n < 0) {
//			nagative = true;
//			if (n == Integer.MIN_VALUE) {
//				n = Integer.MAX_VALUE;
//				x = basicX * x;
//			}
//		}
//		while (n != 1) {
//			if (n % 2 == 1) {
//				x = basicX * x;
//				n--;
//			}
//			x *= x;
//			n /= 2;
//		}
//		return nagative ? 1 / x : x;


		if (n == 0) {
			return 1D;
		}
		if (n < 0) {
			return 1 / x * myPow(1 / x, -(n + 1));
		}
		return n % 2 == 0 ? myPow(x * x, n / 2) : x * myPow(x * x, (n - 1) / 2);
	}

	// ===========50=========
	// ============169=========
	public int majorityElement(int[] nums) {
		// int onesecond = nums.length / 2;
		// HashMap<Integer, Integer> times = new HashMap();
		// for (int num : nums) {
		// if (times.get(num) != null) {
		// times.put(num, times.get(num) + 1);
		// } else
		// times.put(num, 1);
		//
		// if (times.get(num) > onesecond) {
		// return num;
		// }
		// }
		// return -1;

		// Arrays.sort(nums);
		// return nums[nums.length / 2];

		// int count = 0, ret = 0;
		// for (int num : nums) {
		// if (count == 0)
		// ret = num;
		// if (num != ret)
		// count--;
		// else
		// count++;
		// }
		// return ret;

		return devidAndConquer(nums, 0, nums.length - 1);
	}

	int devidAndConquer(int[] nums, int start, int end) {
		if (start == end) {
			return nums[start];
		}
		int mid = (end - start) / 2 + start;

		int left = devidAndConquer(nums, start, mid);
		int right = devidAndConquer(nums, mid + 1, end);

		if (left == right) {
			return left;
		}
		int leftCount = countItems(nums, left, start, end);
		int rightCount = countItems(nums, right, start, end);

		return leftCount > rightCount ? left : right;
	}

	int countItems(int[] nums, int num, int start, int end) {
		int count = 0;
		for (int i = start; i <= end; i++) {
			if (num == nums[i]) {
				count++;
			}
		}
		return count;
	}

	// ============169=========
	// =============22=========??
	public List<String> generateParenthesis(int n) {
		List<String> ans = new ArrayList();
		backtrack(ans, "", n, n);
		return ans;
	}

	public void backtrack(List<String> ans, String cur, int open, int close) {
		if (open == 0 && close == 0) {
			ans.add(cur);
			return;
		}
		if (open > 0) {
			backtrack(ans, cur + "(", open - 1, close);
		}
		if (close > open) {
			backtrack(ans, cur + ")", open, close - 1);
		}
	}

	// =============22=========
	// ==========122============
	public int maxProfit(int[] prices) {
//		int profit = 0;
//		for (int i = 0; i < prices.length - 1; i++) {
//			if (prices[i] < prices[i + 1]) {
//				profit = profit + prices[i + 1] - prices[i];
//			}
//		}
//		return profit;


		int profit = 0;
		for (int i = 0; i < prices.length; i++) {
			if (prices[i] < prices[i + 1]) {
				profit += prices[i + 1] - prices[i];
			}
		}
		return profit;
	}

	// ==========122============
	// ===========51=============???
	public List<List<String>> solveNQueens(int n) {
		if (n < 1) {
			return new ArrayList();
		}
		Set<Integer> col = new HashSet();
		Set<Integer> pie = new HashSet();
		Set<Integer> na = new HashSet();
		List<Stack<Integer>> result = new ArrayList();
		_dfsQueens(n, result, 0, col, pie, na, new Stack<Integer>());

		return printQueens(result, n);
	}

	List<List<String>> printQueens(List<Stack<Integer>> result, int n) {
		System.out.println(result);

		List<List<String>> resultQueens = new ArrayList();
		// for (Stack<Integer> singelLine : result) {
		// List<String> resultLine = new ArrayList();
		// int index = singelLine.pop();
		// for (int i = 0; i< n;i++)) {
		//
		// if (i == ) {
		//
		// }else {
		//
		// }
		// }
		// }
		return resultQueens;
	}

	void _dfsQueens(int n, List<Stack<Integer>> result, int row, Set<
			Integer> col, Set<Integer> pie, Set<Integer> na, Stack<Integer> state) {
		if (row >= n) {
			result.add((Stack<Integer>) state.clone());
			return;
		}

		for (int i = 0; i < n; i++) {
			if (col.contains(i) || pie.contains(row + i) || na.contains(row - i)) {
				continue;
			}

			col.add(i);
			pie.add(row + i);
			na.add(row - i);

			state.add(i);// should use an array instead of list reference
			_dfsQueens(n, result, row + 1, col, pie, na, state);

			col.remove(i);
			pie.remove(row + i);
			na.remove(row - i);
		}
	}

	// ===========51=============
	// ============36=============
	public boolean isValidSudoku(char[][] board) {
		boolean good = sudokuHelper(board);
		for (int i = 0; i < board.length; i++) {
			for (char j : board[i]) {
				System.out.print(j);
			}
			System.out.println();
		}
		return good;
	}

	boolean sudokuHelper(char[][] board) {
		for (int i = 0; i < board.length; i++)
			for (int j = 0; j < board[i].length; j++) {
				if (board[i][j] == '.') {
					for (char k = '1'; k <= '9'; k++) {
						if (validSudokuK(board, i, j, k)) {
							board[i][j] = k;
							if (sudokuHelper(board))
								return true;
							else
								board[i][j] = '.';
						}
					}
					return false;
				}
			}
		return true;
	}


	private boolean validSudokuK(char[][] board, int row, int col, char k) {
		for (int p = 0; p < board.length; p++) {
			if (board[row][p] != '.' && board[row][p] == k) {
				return false;
			}
			if (board[p][col] != '.' && board[p][col] == k) {
				return false;
			}
			if (board[3 * (row / 3) + p / 3][3 * (col / 3) + p % 3] != '.' && board[3 * (row / 3) + p / 3][3 * (col / 3) + p % 3] == k) {
				return false;
			}
		}

		return true;
	}

	// ============36=============
	// ===========102============
	public List<List<Integer>> levelOrder(TreeNode root) {
		if (root == null) {
			return new ArrayList();
		}
		// List<List<Integer>> result = new ArrayList();
		// LinkedList<TreeNode> q = new LinkedList();
		// q.add(root);
		// // Set<TreeNode> visited = new HashSet();18526083910
		// while (!q.isEmpty()) {
		// int levelSize = q.size();
		// List<Integer> currentLevel = new ArrayList();
		//
		// for (int i = 0; i < levelSize; i++) {
		// TreeNode currentNode = q.pollFirst();
		// currentLevel.add(currentNode.val);
		// if (currentNode.left != null) {
		// q.add(currentNode.left);
		// }
		// if (currentNode.right != null) {
		// q.add(currentNode.right);
		// }
		// }
		// result.add(currentLevel);
		// }
		// return result;

		List<List<Integer>> result = new ArrayList();
		dfs(result, root, 0);
		return result;
	}

	void dfs(List<List<Integer>> result, TreeNode node, int level) {
		if (node == null)
			return;
		if (result.size() < level + 1) {
			result.add(new ArrayList());
		}

		result.get(level).add(node.val);
		dfs(result, node.left, level + 1);
		dfs(result, node.right, level + 1);
	}

	// ===========102============
	// ==========104============
	public int maxDepth(TreeNode root) {
		if (root == null) {
			return 0;
		}
		return 1 + Math.max(maxDepth(root.left), maxDepth(root.right));
	}

	// ==========104============
	// ===========111============
	public int minDepth(TreeNode root) {
		if (root == null)
			return 0;
		if (root.left == null)
			return 1 + minDepth(root.right);
		if (root.right == null)
			return 1 + minDepth(root.left);
		return 1 + Math.min(minDepth(root.left), minDepth(root.right));
	}

	// ===========111============
	// ==========20===========
	public boolean isValid(String s) {
		if (s == null || s.trim().equals("")) {
			return true;
		}
		HashMap<String, String> pairs = new HashMap<String, String>();
		pairs.put(")", "(");
		pairs.put("]", "[");
		pairs.put("}", "{");

		Stack<String> brackets = new Stack<String>();
		for (int i = 0; i < s.length(); i++) {
			if (pairs.containsValue(s.substring(i, i + 1))) {
				brackets.push(s.substring(i, i + 1));
			} else {
				if (brackets.isEmpty())
					return false;
				if (!brackets.pop().equals(pairs.get(s.substring(i, i + 1))))
					return false;
			}
		}
		if (brackets.isEmpty()) {
			return true;
		} else {
			return false;
		}
	}
	// ==========20===========

	int devidedSearch(int[] input, int node) {
		if (input == null || input.length <= 0) {
			return -1;
		}
		Arrays.sort(input);
		int mid = input.length / 2;
		int left = 0, right = input.length - 1;
		while (left <= right) {
			mid = (left + right) / 2;
			if (input[mid] == node) {
				return mid;
			}
			if (input[mid] > node) {
				right = mid - 1;
			}
			if (input[mid] < node) {
				left = mid + 1;
			}
		}
		return -1;
	}

	// =============56===========
	public int mySqrt(int x) {
//		if (x <= 1) {
//			return x;
//		}
//		int l = 1;
//		int r = x;
//		int mid = 1;
//		while (l <= r) {
//			mid = l + ((r - l) >> 1);
//			if (mid == x / mid) return mid;
//			if (mid < x / mid) l = mid + 1;
//			if (mid > x / mid) r = mid - 1;
//		}
//		return r;

		if (x <= 1) {
			return x;
		}
		int r = x;
		while (r > x / r) {
			r = (r + x / r) / 2;
		}
		return r;
	}

	// =============56===========

	// ===========191==============
	// you need to treat n as an unsigned value
	public int hammingWeight(int n) {
//		int x = 0;
//		while (n != 0) {
//			n = n & (n - 1);
//			x++;
//		}
//		return x;

		int index = 32;
		int count = 0;
		while (index > 0) {
			if ((n & 1) == 1) {
				count++;
			}
			n >>= 1;
			index--;
		}

		return count;
	}
	// ===========191===============

	// ===========231=============
	public boolean isPowerOfTwo(int n) {
		return n > 0 && ((n & (n - 1)) == 0);
	}
	// =============231==========


	// =============338==========
	public int[] countBits(int num) {
		int[] counts = new int[num + 1];
		counts[0] = 0;
		for (int i = 1; i <= num; i++) {
			int count = 0;
			int temp = i;
			while (temp != 0) {
				temp = temp & (temp - 1);
				count++;
			}
			counts[i] = count;
		}
		return counts;

//
//		int[] counts = new int[num + 1];
//		for (int i = 1; i <= num; i++) {
//			counts[i] += counts[(i & (i - 1))] + 1;
//		}
//		return counts;
	}
	// =============338==========


	// =============70============
	public int climbStairs(int n) {
		return -1;
	}
	// =============70============


	// =============152============
	public int maxProduct(int[] nums) {
		if (nums == null || nums.length == 0) {
			return 0;
		}
		if (nums.length == 1)
			return nums[0];

		int result = nums[0];
		int currentMax = nums[0];
		int currentMin = nums[0];

		for (int i : nums) {
			currentMax = currentMax * i;
			currentMin = currentMin * i;
			currentMax = Math.max(Math.max(currentMax, currentMin), i);
			currentMin = Math.min(Math.min(currentMax, currentMin), i);
			result = currentMax > result ? currentMax : result;
		}

		return result;
	}
	// =============152============
}

class ListNode {
	int val;
	ListNode next;

	ListNode(int x) {
		val = x;
	}
}

// ===========232==============
class MyQueue {
	Stack<Integer> queue;

	/**
	 * Initialize your data structure here.
	 */
	public MyQueue() {
		queue = new Stack();
	}

	/**
	 * Push element x to the back of queue.
	 */
	public void push(int x) {
		Stack<Integer> tempStack = new Stack();
		while (!queue.isEmpty()) {
			tempStack.push(queue.pop());
		}
		tempStack.push(x);
		while (!tempStack.isEmpty()) {
			queue.push(tempStack.pop());
		}
	}

	/**
	 * Removes the element from in front of queue and returns that element.
	 */
	public int pop() {
		if (!queue.isEmpty()) {
			return queue.pop();
		}
		return -1;
	}

	/**
	 * Get the front element.
	 */
	public int peek() {
		if (!queue.isEmpty()) {
			return queue.peek();
		}
		return -1;

	}

	/**
	 * Returns whether the queue is empty.
	 */
	public boolean empty() {
		return queue.isEmpty();
	}
}

/**
 * Your MyQueue object will be instantiated and called as such: MyQueue obj = new MyQueue(); obj.push(x); int param_2 = obj.pop(); int param_3 = obj.peek(); boolean param_4 =
 * obj.empty();
 */

// ===========232==============

// ===============225=========
class MyStack {
	Queue<Integer> stack = null;
	int items = 0;
	int top = -1;

	/**
	 * Initialize your data structure here.
	 */
	public MyStack() {
		stack = new LinkedList();
	}

	/**
	 * Push element x onto stack.
	 */
	public void push(int x) {
		stack.add(x);
		items++;
		top = x;
	}

	/**
	 * Removes the element on top of the stack and returns that element.
	 */
	public int pop() {
		int counts = 1;
		int result = -1;
		Queue<Integer> stackTemp = new LinkedList();
		while (!stack.isEmpty()) {
			if (counts == items) {
				result = stack.poll();
			} else {
				int loopitem = stack.poll();
				stackTemp.add(loopitem);
				top = loopitem;
			}
			counts++;
		}
		stack = stackTemp;
		items--;
		return result;
	}

	/**
	 * Get the top element.
	 */
	public int top() {
		return top;
	}

	/**
	 * Returns whether the stack is empty.
	 */
	public boolean empty() {
		return stack.isEmpty();
	}
}

/**
 * Your MyStack object will be instantiated and called as such: MyStack obj = new MyStack(); obj.push(x); int param_2 = obj.pop(); int param_3 = obj.top(); boolean param_4 =
 * obj.empty();
 */

// ==============225==========
// ===============703=========
class KthLargest {
	PriorityQueue<Integer> knumbers = null;
	int k = 0;

	public KthLargest(int k, int[] nums) {
		this.k = k;
		knumbers = new PriorityQueue(k);
		for (int i = 0; i < nums.length; i++) {
			add(nums[i]);
		}
	}

	public int add(int val) {
		if (knumbers.size() < this.k) {
			knumbers.add(val);
		} else if (val > knumbers.peek()) {
			knumbers.poll();
			knumbers.add(val);
		}
		return knumbers.peek();
	}
}
// ========703===========

// ========binary search=======
class BinarySearch {
	int doSearch(int[] i, int t) {
		int left = 0;
		int right = i.length;
		while (left <= right) {
			int mid = left + (right - left) >> 1;
			if (i[mid] == t) return mid;
			if (t < i[mid]) right = mid - 1;
			if (t > i[mid]) left = mid + 1;
		}
		return -1;
	}
}

//=========union and find===========
class UnionAndFind {
	int[] father = null;
	public int findFather(int node) {
		if (father[node] == node)
			return node;
		return father[node] = findFather(father[node]);
	}

	public void union(int a, int b) {
		int aFather = findFather(a);
		int bFather = findFather(b);

		if (father[aFather] != father[bFather])
			father[bFather] = father[aFather];
	}
}


class ConnectingGraph3 {
	/**
	 * @param a: An integer
	 * @param b: An integer
	 * @return: nothing
	 */

	int[] father;
	int count = 0;
	public ConnectingGraph3(int n) {

		// initialize your data structure here.
		father = new int[n+1];
		count = n;
		for (int i = 1; i <= n; i++) {
			father[i] = i;
		}
	}

	public void connect(int a, int b) {
		// write your code here
		int aFather = find(a);
		int bFather = find(b);
		if (aFather != bFather) {
			father[aFather] = bFather;
			count--;
		}
	}

	public int find(int node) {
		if (father[node] == node)
			return node;
		return father[node] = find(father[node]);
	}

	/**
	 * @return: An integer
	 */
	public int query() {
		// write your code here
		return count;
	}
}