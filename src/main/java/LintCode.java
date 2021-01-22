import java.util.*;


public class LintCode {

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


    public String minWindow(String source, String target) {
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

            if (c - 1 == k || c == k) {
                if (finalLeft == -1 || finalRight - finalLeft < (right - left)) {
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
            return kthSmallest(k, Arrays.copyOfRange(nums, 0, pivotIndex));
        } else {
            return kthSmallest(k - pivotIndex - 1, Arrays.copyOfRange(nums, pivotIndex + 1, nums.length));
        }
    }

    int partitioning(int[] nums, int start, int end) {
        int pivot = nums[start];
        int i = start;
        int j = end;
        while (i < j) {
            while (i < j && pivot <= nums[--j]) ;
            if (i < j) {
                nums[i] = nums[j];
            }
            while (i < j && pivot >= nums[++i]) ;
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
        for (int i = 0; i < arrays.length; i++) {
            for (int j = 0; j < arrays[i].length; j++) {
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
            if (next_y < n && !used[current_x][next_y]) {
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
        public int compare(Sum a, Sum b) {
            return a.val - b.val;
        }
    }

    public int kthSmallestSum(int[] A, int[] B, int kk) {
        // write your code here
        PriorityQueue<Sum> pq = new PriorityQueue<Sum>(kk, new SumComparator());

        Sum sum = new Sum(0, 0, A[0] + B[0]);
        pq.offer(sum);

        int[] dx = new int[]{1, 0};
        int[] dy = new int[]{0, 1};
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

    int[] twoToOne(int[][] two) {
        int[] result = new int[two.length * two[0].length];
        for (int i = 0; i < two.length; i++) {
            for (int j = 0; j < two[i].length; j++) {
                result[(i * two[i].length) + j] = two[i][j];
            }
        }
        return result;
    }

    int[][] oneToTow(int[] one) {
        int[][] result = new int[one.length / 4][one.length / 4];
        for (int i = 0; i < one.length; i++) {
            result[i / result.length][i % result[0].length] = one[i];
        }
        return result;
    }


    public List<Integer> numIslands2(int n, int m, Point[] operators) {
        if (n <= 0 || m <= 0 || operators == null) return null;

        // write your code here
        int[] dx = new int[]{1, 0, -1, 0};
        int[] dy = new int[]{0, 1, 0, -1};

        int[][] ocean = new int[n][m];
        father = new int[n * m];
        for (int i = 0; i < n * m; i++) {
            father[i] = i;
        }

        int count = 0;
        List<Integer> result = new ArrayList<Integer>();

        for (Point p : operators) {
            if (ocean[p.x][p.y] == 0) {
                ocean[p.x][p.y] = 1;
                count++;

//				System.out.println();
//				for (int[] oo : ocean) {
//					for (int o : oo) {
//						System.out.print(o + "，");
//					}
//					System.out.println();
//				}
//				for (int i = 0; i < father.length ; i++) {
//					System.out.print(i + "["+father[i]+"]");
//				}
//				System.out.println();

                for (int i = 0; i < 4; i++) {
                    int x = p.x + dx[i];
                    int y = p.y + dy[i];
                    if (x >= 0 && y >= 0 && x < n && y < m) {
                        if (ocean[x][y] == 1) {
                            if (find(father[x * m + y]) != find(father[p.x * m + p.y])) {
                                union(father[p.x * m + p.y], father[x * m + y]);
                                count--;
                            }
                        }
                    }
                }
            }
            result.add(count);
        }
        return result;
    }

    int[] father;

    int find(int a) {
        if (father[a] == a)
            return a;
        return father[a] = find(father[a]);
    }

    void union(int aa, int bb) {
        int a = find(aa);
        int b = find(bb);
        if (a != b)
            father[b] = a;
    }


    public List<List<String>> wordSquares(String[] words) {
        // write your code here
        List<List<String>> wordsResult = new ArrayList();

        List<String> wordsH = new ArrayList();
        TrieLintCode t = new TrieLintCode();
        for (String word : words) {
            t.insert(word);
            wordsH.add(word);
        }
        wordsResult.add(wordsH);

        List<String> wordsV = new ArrayList();
        for (int i = 0; i < words[0].length(); i++) {
            String word = "";
            for (int j = 0; j < words.length; j++) {
                word += words[j].toCharArray()[i];
            }
            TrieNodeLintCode node = t.search(word);
            if (node != null && node.isEnd) {
                wordsV.add(word);
            }
        }

        wordsResult.add(wordsV);
        return wordsResult;
    }


    public int trapRainWater(int[][] heights) {
        // write your code here
        int res = 0;
        int m = heights.length;
        int n = heights[0].length;
        boolean[][] visited = new boolean[m][n];
        PriorityQueue<Cell> pq = new PriorityQueue(new CellComparator());
        for (int i = 0; i < m; i++) {
            pq.offer(new Cell(i, 0, heights[i][0]));
            visited[i][0] = true;
            pq.offer(new Cell(i, n - 1, heights[i][n - 1]));
            visited[i][n - 1] = true;
        }

        for (int i = 0; i < n; i++) {
            pq.offer(new Cell(0, i, heights[0][i]));
            visited[0][i] = true;
            pq.offer(new Cell(m - 1, i, heights[m - 1][i]));
            visited[m - 1][i] = true;
        }

        int[] dx = new int[]{1, 0, -1, 0};
        int[] dy = new int[]{0, 1, 0, -1};
        while (!pq.isEmpty()) {
            Cell c = pq.poll();

            for (int i = 0; i < 4; i++) {
                int nx = c.x + dx[i];
                int ny = c.y + dy[i];

                if (nx >= 0 && nx < m && ny >= 0 && ny < n && !visited[nx][ny]) {
                    int nh = heights[nx][ny];
                    visited[nx][ny] = true;
                    if (nh < c.height) {
                        res += c.height - nh;
                        pq.offer(new Cell(nx, ny, c.height));
                    } else {
                        pq.offer(new Cell(nx, ny, nh));
                    }
                }
            }
        }

        return res;
    }


    PriorityQueue<Integer> minHeap = new PriorityQueue();
    PriorityQueue<Integer> maxHeap = new PriorityQueue(Collections.reverseOrder());
    int count = 0;

    public int[] medianII(int[] nums) {
        // write your code here
        int[] result = new int[nums.length];
        for (int i = 0; i < nums.length; i++) {
            addNumber(nums[i]);
            result[i] = getNumber();
        }

        return result;
    }

    void addNumber(int num) {
        maxHeap.offer(num);
        if (count % 2 == 0) {
            if (minHeap.isEmpty()) {
                count++;
                return;
            } else if (maxHeap.peek() > minHeap.peek()) {
                int max = maxHeap.poll();
                int min = minHeap.poll();
                maxHeap.offer(min);
                minHeap.offer(max);
            }
        } else {
            minHeap.offer(maxHeap.poll());
        }
        count++;
    }

    int getNumber() {
        return maxHeap.peek();
    }


    TreeSet<Node> max = new TreeSet();
    TreeSet<Node> min = new TreeSet();

    public List<Integer> medianSlidingWindow(int[] nums, int k) {
        // write your code here

        if (k <= 0) {
            return null;
        }
        List<Integer> res = new ArrayList();
        if (k == 1) {
            res.add(nums[0]);
            return res;
        }

        int half = (k + 1) >> 1;
        for (int i = 0; i < k - 1; i++) {
            addNumber(new Node(i, nums[i]), half);
        }

        for (int i = k - 1; i < nums.length; i++) {
            addNumber(new Node(i, nums[i]), half);
            res.add(max.first().val);
            removeNumber(new Node(i - k + 1, nums[i - k + 1]));
        }
        return res;
    }

    void addNumber(Node node, int half) {
        if (max.size() < half) {
            max.add(node);
        } else {
            min.add(node);
        }

        if (max.size() == half && min.size() > 0) {
            if (min.last().val > max.first().val) {
                Node last = min.last();
                Node first = max.first();
                min.remove(last);
                max.remove(first);
                min.add(first);
                max.add(last);
            }
        }
    }

    void removeNumber(Node node) {
        if (min.contains(node)) {
            min.remove(node);
        } else {
            max.remove(node);
        }
    }


    Stack<Object> stack = new Stack();

    public String expressionExpand(String s) {
        // write your code here
        int number = 0;
        for (char c : s.toCharArray()) {
            if (Character.isDigit(c)) {
                number = number * 10 + c - '0';
            } else if (c == '[') {
                stack.push(number);
                number = 0;
            } else if (c == ']') {
                String subString = popStack();
                Integer times = (Integer) stack.pop();
                String subStringResult = "";
                for (int j = 0; j < times; j++) {
                    subStringResult += subString;
                }
                stack.push(subStringResult);
            } else {
                stack.push(String.valueOf(c));
            }
        }

        for (char c : popStack().toCharArray()) {
            stack.push(String.valueOf(c));
        }
        return popStack();
    }

    String popStack() {
        String result = "";
        while (!stack.isEmpty() && stack.peek() instanceof String) {
            result += stack.pop();
        }

        return result;
    }


    class Record {
        public int id, score;

        public Record(int id, int score) {
            this.id = id;
            this.score = score;
        }
    }


    class ScoreComparator implements Comparator<Record> {
        public int compare(Record a, Record b) {
            return b.score - a.score;
        }
    }

    public Map<Integer, Double> highFive(Record[] results) {
        // Write your code here
        Map<Integer, PriorityQueue> student = new HashMap();
        for (Record record : results) {
            if (student.get(record.id) == null) {
                PriorityQueue<Record> pq = new PriorityQueue(5, new ScoreComparator());
                pq.offer(record);
                student.put(record.id, pq);
            } else {
                PriorityQueue<Record> pq = student.get(record.id);
                pq.offer(record);
            }
        }

        Map<Integer, Double> result = new HashMap();
        for (int id : student.keySet()) {
            double scoreTotal = 0;
            for (int i = 0; i < 5; i++) {
                scoreTotal += ((Record)student.get(id).poll()).score;
            }
            result.put(id, scoreTotal / 5d);
        }
        return result;
    }

    public int largestRectangleArea(int[] heights) {
        // write your code here

        int squareMax = 0;
        for (int i = 0; i< heights.length; i++) {
            int heightMin = Integer.MAX_VALUE;
            for (int j = i; j< heights.length; j++) {
                heightMin = Math.min(heightMin, Math.min(heights[i], heights[j]));
                squareMax = Math.max(squareMax, heightMin * (j - i + 1));
            }
        }

        return squareMax;
    }



    public TreeNode maxTree(int[] A) {
        if (A == null || A.length <= 0) {
            return null;
        }
        int maxIndex = 0;
        for (int i = 0; i < A.length; i++) {
            if (A[maxIndex] <  A[i]) {
                maxIndex = i;
            }
        }
        TreeNode maxNode = new TreeNode(A[maxIndex]);

        maxNode.left = maxTree(Arrays.copyOfRange(A, 0, maxIndex));
        maxNode.right = maxTree(Arrays.copyOfRange(A, maxIndex + 1, A.length));

        return maxNode;
    }

    public static void main(String[] args) {

        LintCode lc = new LintCode();

        System.out.println(lc.lengthOfLongestSubstring("abcabcbb"));
        System.out.println(lc.minWindow("aaaaaaaaaaaabbbbbcdd", "abcdd"));
        System.out.println(lc.lengthOfLongestSubstringKDistinct("igtpevzimytyukifgezynnksysssnohespcwiqpheetgjtgmxkeqqoxldqkribsrkmooiyqkpjxaxllmizwiqzribq", 17));
        System.out.println(lc.kthSmallest(10, new int[]{1, 2, 3, 4, 5, 6, 8, 9, 10, 7}));
        System.out.println(lc.kthSmallest(new int[][]{
                {1, 3, 5, 7, 9},
                {2, 4, 6, 8, 10},
                {11, 13, 15, 17, 19},
                {12, 14, 16, 18, 20},
                {21, 22, 23, 24, 25}}, 8));
        System.out.println(lc.kthSmallestSum(new int[]{1, 7, 11}, new int[]{2, 4, 6}, 3));

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


        System.out.println();
        System.out.println(Arrays.toString(lc.twoToOne(new int[][]{
                {1, 3, 5, 7, 9},
                {2, 4, 6, 8, 10},
                {11, 13, 15, 17, 8},
                {12, 14, 16, 18, 8},
                {21, 22, 23, 24, 8}})));


        System.out.println();
        int[][] result = lc.oneToTow(new int[]{1, 3, 5, 7, 9, 2, 4, 6, 8, 10, 11, 13, 15, 17, 8, 12});
        for (int[] rr : result) {
            for (int r : rr) {
                System.out.print(r + "，");
            }
            System.out.println();
        }


        System.out.println();
//		System.out.println(lc.numIslands2(4, 5, new Point[]{new Point(1,1), new Point(0,1), new Point(3,3), new Point(3,4)}));
//		System.out.println(lc.numIslands2(2, 2, new Point[]{new Point(0,0), new Point(1,1), new Point(1,0), new Point(0,1)}));
        System.out.println(lc.numIslands2(8, 14, new Point[]{
                new Point(0, 9),
                new Point(5, 4),
                new Point(0, 12),
                new Point(6, 9),
                new Point(6, 5),
                new Point(0, 4),
                new Point(4, 11),
                new Point(0, 0),
                new Point(3, 5),
                new Point(6, 7),
                new Point(3, 12),
                new Point(0, 5),
                new Point(6, 13),
                new Point(7, 5),
                new Point(3, 6),
                new Point(4, 4),
                new Point(0, 8),
                new Point(3, 1),
                new Point(4, 6),
                new Point(6, 1),
                new Point(5, 12),
                new Point(3, 8),
                new Point(7, 0),
                new Point(2, 9),
                new Point(1, 4),
                new Point(3, 0),
                new Point(1, 13),
                new Point(2, 13),
                new Point(6, 0),
                new Point(6, 4),
                new Point(0, 13),
                new Point(0, 3),
                new Point(7, 4),
                new Point(1, 8),
                new Point(5, 5),
                new Point(5, 7),
                new Point(5, 10),
                new Point(5, 3),
                new Point(6, 10),
                new Point(6, 2),
                new Point(3, 10),
                new Point(2, 7),
                new Point(1, 12),
                new Point(5, 0),
                new Point(4, 5),
                new Point(7, 13),
                new Point(3, 2)
        }));

        TrieLintCode t = new TrieLintCode();
        t.searchNode("lintcode");
        t.searchNode("lint");
        t.insert("lint");
        t.searchNode("lint");

        System.out.println(lc.wordSquares(new String[]{"area", "lead", "wall", "lady", "ball"}));


        lc.medianII(new int[]{1, 2, 3, 4, 5});


        lc.medianSlidingWindow(new int[]{142, 38, 100, 53, 22, 84, 168, 50, 194, 136, 111, 13, 47, 45, 151, 164, 126, 47, 106, 124, 183, 8, 87, 38, 91, 121, 102, 46, 82, 195, 53, 18, 11, 165, 61}, 35);

        MinStack ms = new MinStack();

        ms.push(23);
        ms.min();
        ms.pop();
        ms.push(24);
        ms.pop();
        ms.push(19);
        ms.min();
        ms.pop();
        ms.push(28);
        ms.min();
        ms.pop();
        ms.push(19);
        ms.min();
        ms.pop();
        ms.push(25);
        ms.min();
        ms.push(26);
        ms.min();
        ms.pop();
        ms.min();
        ms.pop();
        ms.push(20);
        ms.min();
        ms.pop();
        ms.push(26);
        ms.min();
        ms.push(21);
        ms.pop();
        ms.min();
        ms.pop();
        ms.push(21);
        ms.pop();
        ms.push(25);
        ms.pop();
        ms.push(23);
        ms.pop();
        ms.push(21);
        ms.min();
        ms.pop();
        ms.push(21);
        ms.min();
        ms.pop();
        ms.push(27);
        ms.min();
        ms.push(19);
        ms.min();
        ms.pop();
        ms.min();
        ms.pop();
        ms.push(27);
        ms.pop();
        ms.push(28);
        ms.min();
        ms.pop();
        ms.push(21);
        ms.min();
        ms.pop();
        ms.push(27);
        ms.min();
        ms.pop();
        ms.push(22);
        ms.pop();
        ms.push(29);
        ms.min();
        ms.pop();
        ms.push(21);
        ms.pop();
        ms.push(24);
        ms.pop();
        ms.push(24);
        ms.pop();
        ms.push(24);
        ms.pop();
        ms.push(29);
        ms.min();
        ms.pop();
        ms.push(23);
        ms.pop();
        ms.push(20);
        ms.min();
        ms.pop();
        ms.push(24);
        ms.min();
        ms.pop();
        ms.push(24);
        ms.min();
        ms.push(20);
        ms.min();
        ms.pop();
        ms.min();
        ms.push(24);
        ms.min();
        ms.push(24);
        ms.min();
        ms.push(29);
        ms.min();
        ms.pop();
        ms.min();
        ms.pop();
        ms.min();
        ms.pop();
        ms.pop();
        ms.push(22);
        ms.pop();
        ms.push(25);
        ms.min();
        ms.pop();
        ms.push(21);
        ms.min();
        ms.pop();
        ms.push(27);
        ms.min();
        ms.pop();
        ms.push(28);
        ms.min();
        ms.pop();
        ms.push(25);
        ms.pop();
        ms.push(27);
        ms.min();
        ms.push(24);
        ms.pop();
        ms.min();
        ms.pop();
        ms.push(25);
        ms.min();
        ms.push(23);
        ms.min();
        ms.pop();
        ms.min();
        ms.push(21);
        ms.min();
        ms.push(22);
        ms.min();
        ms.push(23);
        ms.min();
        ms.pop();
        ms.min();
        ms.pop();
        ms.min();
        ms.pop();
        ms.min();
        ms.push(25);
        ms.min();
        ms.pop();


        System.out.println(lc.expressionExpand("3[2[ad]3[pf]]xyz"));

        System.out.println(lc.largestRectangleArea(new int[]{1,1,1,1,1}));


        System.out.println(lc.maxTree(new int[]{2,5,6,0,3,1}));
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
        father = new int[n + 1];
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


class Point {
    int x;
    int y;

    Point() {
        x = 0;
        y = 0;
    }

    Point(int a, int b) {
        x = a;
        y = b;
    }
}

class TrieLintCode {
    TrieNodeLintCode root = new TrieNodeLintCode();

    void insert(String word) {
        char[] cc = word.toCharArray();
        Map<Character, TrieNodeLintCode> children = root.children;
        for (int i = 0; i < cc.length; i++) {
            TrieNodeLintCode node = children.get(cc[i]);
            if (node == null) children.put(cc[i], new TrieNodeLintCode());
            if (i == (cc.length - 1)) children.get(cc[i]).isEnd = true;
            children = children.get(cc[i]).children;
        }
    }

    List<TrieNodeLintCode> searchNode(String word) {
        char[] cc = word.toCharArray();

        List<Map<Character, TrieNodeLintCode>> children = new ArrayList();
        children.add(root.children);
        List<TrieNodeLintCode> cur = null;
        for (int i = 0; i < cc.length; i++) {
            char c = cc[i];
            if ('.' == c) {
                cur = new ArrayList();
                for (Map<Character, TrieNodeLintCode> ccc : children) {
                    if (!ccc.values().isEmpty()) {
                        cur.addAll(ccc.values());
                    }
                }
            } else {
                cur = new ArrayList();
                for (Map<Character, TrieNodeLintCode> ccc : children) {
                    if (ccc.containsKey(cc[i])) {
                        cur.add(ccc.get(cc[i]));
                    }
                }
            }
            if (cur == null || cur.size() <= 0) return null;
            if (i == (cc.length - 1)) return cur;
            children = new ArrayList();
            for (TrieNodeLintCode node : cur) {
                children.add(node.children);
            }
        }
        return null;
    }

    TrieNodeLintCode search(String word) {
        char[] c = word.toCharArray();
        Map<Character, TrieNodeLintCode> children = root.children;
        TrieNodeLintCode cur = null;
        for (int i = 0; i < c.length; i++) {
            cur = children.get(c[i]);
            if (cur == null) return null;
        }
        return cur;
    }
}

class TrieNodeLintCode {
    Map<Character, TrieNodeLintCode> children = new HashMap();
    boolean isEnd;
}


class Cell {
    int x;
    int y;
    int height;

    Cell(int x, int y, int height) {
        this.x = x;
        this.y = y;
        this.height = height;
    }
}

class CellComparator implements Comparator<Cell> {
    public int compare(Cell a, Cell b) {
        return a.height - b.height;
    }
}


class Node implements Comparable<Node> {
    int id;
    int val;

    Node(int id, int val) {
        this.id = id;
        this.val = val;
    }

    public int compareTo(Node node) {
        return this.val - node.val;
    }
}


class MinStack {
    Stack<Integer> stack = null;
    Stack<Integer> stackMin = null;
    int min = Integer.MAX_VALUE;

    public MinStack() {
        // do intialization if necessary
        stack = new Stack();
        stackMin = new Stack();
    }

    /*
     * @param number: An integer
     * @return: nothing
     */

    public void push(int number) {
        // write your code here
        stack.push(number);
        min = Math.min(min, number);
        stackMin.push(min);
    }

    /*
     * @return: An integer
     */
    public int pop() {
        // write your code here
        stackMin.pop();
        if (stackMin.isEmpty()) min = Integer.MAX_VALUE;
        else min = stackMin.peek();
        return stack.pop();
    }

    /*
     * @return: An integer
     */
    public int min() {
        // write your code here
        return stackMin.peek();
    }
}