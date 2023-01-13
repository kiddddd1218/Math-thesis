# Some problems I have:
1. As you can see in the notebook, I tried to implement projected bfgs through two different ways: iterative and recursion. In the book, it gave a pesudocode of recursive manner. I tried this, but it seems impossible for the recursive implementation to terminate when the number of nodes is above 50. 
2. So I tried to implement this algorithm using only iterations. And it works fine if enough time is given. As you can see in the notebook, this implementation requires 1 min to caculate 99 nodes...
3. I am not sure if there is some problem that I cannot see myself in my implementations. It would be so great if you can check my implementations of algorithms in 5.5.3.
4. One more thing... For all optimization algorithms I tried so far, I used them in two different structures: To optimize (xs,ws) as a whole or to fix xs and solve nnls for ws, then apply optimization algorithms on xs till converge.   
