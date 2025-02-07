# Ideas for the Replay Buffer Analysis

1. Problem of the normal ER: Higher P for all early experiences
2. Balance based on memory decay (BER): Idea long term experiences are less important **TODO**
3. Prioritize memory (PER): Idea of choosing high td error or high rewards
   - **Problems**:
     - PER converges around 0 - 5.
     - The buffer is full after step 5k ?! Makes no sense. Try other buffers to validate behavior.
   - **New Thoughts**:
     - Scale the error/reward to be more stable
     - Use bigger memory
   - **Ideas**:
     - Run as a SumTree -> Sample and Insert O(log n)
     - Run as Array/Heap Structure -> Sample O(n) and Insert O(1)

4. Combine Balanced and Prioritized (BPER): Idea to combine all thoughts **TODO**
5. Add strong negative reward if I don't hit the ball at all the whole game.
