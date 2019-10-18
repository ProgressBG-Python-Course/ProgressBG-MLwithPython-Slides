from random import randint
import math, time



class GuessTheNumber:
  """NOTE

    Next code tries to be as simple as possible and as close as possible to the procedural programming.
    The aim is to be readable for non OOP programmers (to illustrate the algorithm).

    The class is used just to escape the need of "global" variables and syntax.

    IT IS NOT AN EXAMPLE OF GOOD PYTHON PROGRAMMING STYLE!
  """

  def __init__(self):
    self.start = int(input("Enter the Start of the interval: "))
    self.end = int(input("Enter the End of the interval: "))
    self.maxTries = math.ceil( math.log2(self.end-self.start+1) )
    self.pastTryNumber = None
    self.newGame()

  def newGame(self):
    self.tryCount = 0

    print("\nNow, think of a number between [{}-{}] and remember it!".format(self.start, self.end))
    time.sleep(1)

    print("\nI'll guess it in max {} tries!".format(self.maxTries))
    ready = input("Are you ready to bet some 🍺 🍺 🍺?[y/n]:")

    if ready in ["n", "N"]:
      print("Ok, looser! Bye!")
      exit()
    else:
      print("""\nI'll ask you some questions. Please, answer me with:
      'C' or 'c' for Correct,
      'U' or 'u' for Up,
      'D' or 'd' for Down
      """)
      self.gameMove(self.start, self.end)

  def gameMove(self, start, end):
    if self.tryCount == self.maxTries :
      self.loseExit()
    self.tryCount += 1

    machineNumber = start + int((end - start)/2)

    userAnswer = self.askUser(machineNumber)
    if userAnswer in ["C", "c"]:
      self.machineVictory()
      self.proposeNewGame()

    elif userAnswer in ["U", "u"]:
      start = machineNumber+1
      self.gameMove(start, end)
    elif userAnswer in ["D","d"]:
      end = machineNumber - 1
      self.gameMove(start, end)

  def askUser(self,machineNumber):
    while(True):
      answer = input("What about {}? :".format(machineNumber))

      if(answer in ["C", "c", "U", "u", "D","d"]):
        return answer
      else:
        print("\tPlease, answer me with 'C|c' or 'U|u' or 'D|d'\n")

  def proposeNewGame(self):
    a = input("\nAnother game?[y/n]")
    if a in ["n", "N"]:
      print("Ok! Bye!")
      exit()
    else:
      self.newGame()

  def machineVictory(self):
    print("\n")
    print('🍺 '*17)
    print("🍺  Hurray! I guessed from {} try! 🍺".format(self.tryCount))
    print('🍺 '*17)

  def loseExit(self):
    print("\n")
    print("👿 " * 22)
    print("👿   It is not possible for me to loose.   👿")
    print("👿   You have mistake! Check your answers! 👿")
    print("👿 " * 22)
    exit();

game = GuessTheNumber()