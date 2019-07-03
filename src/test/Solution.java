package test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Set;
import java.util.Stack;
import java.util.concurrent.CountDownLatch;
import java.util.regex.Matcher;

import javax.swing.text.AbstractDocument.LeafElement;
import javax.swing.text.html.StyleSheet.BoxPainter;
import javax.xml.bind.ValidationEvent;

class TreeNode {
	int val;
	TreeNode left;
	TreeNode right;

	TreeNode(int x) {
		val = x;
	}
}

public class Solution {

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
		List<TreeNode> pathP = new ArrayList<>();
		List<TreeNode> pathQ = new ArrayList<>();

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
		if (root == null || root == p || root == q)
			return root;
		TreeNode left = lowestCommonAncestor(root.left, p, q);
		TreeNode right = lowestCommonAncestor(root.right, p, q);
		return left == null ? right : right == null ? left : root;
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
		HashMap<Integer, Integer> hash_map = new HashMap<>();
		for (int i = 0; i < nums.length; i++) {
			if (hash_map.containsKey(target - nums[i])) {
				return new int[] { i, hash_map.get(target - nums[i]) };
			} else {
				hash_map.put(nums[i], i);
			}
		}
		return null;
	}
	// =================1================

	// ===========15=============
	public List<List<Integer>> threeSum(int[] nums) {
		// List<List<Integer>> result = new ArrayList<>();
		// HashSet<Integer> c = new HashSet<>();
		// for (int i = 0; i < nums.length; i++) {
		// c.add(nums[i]);
		// }
		// for (int i = 0; i < nums.length; i++) {
		// for (int j = i; j < nums.length; j++) {
		// if (c.contains(0 - nums[i] - nums[j])) {
		// List<Integer> resultInner = new ArrayList<>();
		// resultInner.add(nums[i]);
		// resultInner.add(nums[j]);
		// resultInner.add(0 - nums[i] - nums[j]);
		// result.add(resultInner);
		// }
		// }
		// }
		//
		List<List<Integer>> result = new ArrayList<>();
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
					List<Integer> resultInner = new ArrayList<>();
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
		List<List<Integer>> result = new ArrayList<>();

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
						List<Integer> resultrow = new ArrayList<>();
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

	// =========239===========
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
			return new int[] {};
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

		// HashMap<String, Integer> setS1 = new HashMap<>();
		// HashMap<String, Integer> setS2 = new HashMap<>();
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
		HashSet<Integer> values = new HashSet<>();
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

		System.out.println(new Solution().mySqrt(8));
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

		if (n == 0) {
			return 1;
		}
		boolean nagative = false;
		double basicX = x;
		if (n < 0) {
			nagative = true;
			if (n == Integer.MIN_VALUE) {
				n = Integer.MAX_VALUE;
				x = basicX * x;
			}
		}
		while (n != 1) {
			if (n % 2 == 1) {
				x = basicX * x;
				n--;
			}
			x *= x;
			n /= 2;
		}
		return nagative ? 1 / x : x;
	}

	// ===========50=========
	// ============169=========
	public int majorityElement(int[] nums) {
		// int onesecond = nums.length / 2;
		// HashMap<Integer, Integer> times = new HashMap<>();
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
		List<String> ans = new ArrayList<>();
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
		int profit = 0;
		for (int i = 0; i < prices.length - 1; i++) {
			if (prices[i] < prices[i + 1]) {
				profit = profit + prices[i + 1] - prices[i];
			}
		}
		return profit;
	}

	// ==========122============
	// ===========51=============???
	public List<List<String>> solveNQueens(int n) {
		if (n < 1) {
			return new ArrayList<>();
		}
		Set<Integer> col = new HashSet<>();
		Set<Integer> pie = new HashSet<>();
		Set<Integer> na = new HashSet<>();
		List<Stack<Integer>> result = new ArrayList<>();
		_dfsQueens(n, result, 0, col, pie, na, new Stack<Integer>());

		return printQueens(result, n);
	}

	List<List<String>> printQueens(List<Stack<Integer>> result, int n) {
		System.out.println(result);

		List<List<String>> resultQueens = new ArrayList<>();
		// for (Stack<Integer> singelLine : result) {
		// List<String> resultLine = new ArrayList<>();
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

	void _dfsQueens(int n, List<Stack<Integer>> result, int row, Set<Integer> col, Set<Integer> pie, Set<Integer> na, Stack<Integer> state) {
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
			return new ArrayList<>();
		}
		// List<List<Integer>> result = new ArrayList<>();
		// LinkedList<TreeNode> q = new LinkedList<>();
		// q.add(root);
		// // Set<TreeNode> visited = new HashSet<>();18526083910
		// while (!q.isEmpty()) {
		// int levelSize = q.size();
		// List<Integer> currentLevel = new ArrayList<>();
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

		List<List<Integer>> result = new ArrayList<>();
		dfs(result, root, 0);
		return result;
	}

	void dfs(List<List<Integer>> result, TreeNode node, int level) {
		if (node == null)
			return;
		if (result.size() < level + 1) {
			result.add(new ArrayList<>());
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
		if (root == null) {
			return 0;
		}
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
		if (x <= 0) {
			return 0;
		}
		if (x == 1) {
			return 1;
		}
		int mid, left = 1, right = x;
		while (left <= right) {
			mid = left + (right - left) / 2;
			if (mid == x / mid) {
				return mid;
			}
			if (mid < x / mid) {
				left = mid + 1;
			} else {
				right = mid - 1;
			}
		}
		return right;
	}
	// =============56===========
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

	/** Initialize your data structure here. */
	public MyQueue() {
		queue = new Stack<>();
	}

	/** Push element x to the back of queue. */
	public void push(int x) {
		Stack<Integer> tempStack = new Stack<>();
		while (!queue.isEmpty()) {
			tempStack.push(queue.pop());
		}
		tempStack.push(x);
		while (!tempStack.isEmpty()) {
			queue.push(tempStack.pop());
		}
	}

	/** Removes the element from in front of queue and returns that element. */
	public int pop() {
		if (!queue.isEmpty()) {
			return queue.pop();
		}
		return -1;
	}

	/** Get the front element. */
	public int peek() {
		if (!queue.isEmpty()) {
			return queue.peek();
		}
		return -1;

	}

	/** Returns whether the queue is empty. */
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

	/** Initialize your data structure here. */
	public MyStack() {
		stack = new LinkedList<>();
	}

	/** Push element x onto stack. */
	public void push(int x) {
		stack.add(x);
		items++;
		top = x;
	}

	/** Removes the element on top of the stack and returns that element. */
	public int pop() {
		int counts = 1;
		int result = -1;
		Queue<Integer> stackTemp = new LinkedList<>();
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

	/** Get the top element. */
	public int top() {
		return top;
	}

	/** Returns whether the stack is empty. */
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
		knumbers = new PriorityQueue<>(k);
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
