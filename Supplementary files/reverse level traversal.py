# Python program to print REVERSE level order traversal using
# stack and queue

from collections import deque

class Node:
	def __init__(self, data):
		self.data = data
		self.left = None
		self.right = None


# Given a binary tree, print its nodes in reverse level order
def reverseLevelOrder(root):
	q = deque()
	q.append(root)
	ans = deque()
	while q:
		node = q.popleft()
		if node is None:
			continue

		ans.appendleft(node.data)

		if node.right:
			q.append(node.right)

		if node.left:
			q.append(node.left)

	return ans


root = Node(1)
root.left = Node(2)
root.right = Node(3)
root.left.left = Node(4)
root.left.right = Node(5)
root.right.left = Node(6)
root.right.right = Node(7)

print ("Level Order traversal of binary tree is")
deq = reverseLevelOrder(root)
for key in deq:
	print (key),
