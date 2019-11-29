# Страничка курса "Обучение с подкреплением и нейронные сети"

Курс посвящен методам обучения с подкреплением (Reinforcement learning) - одному из способов машинного обучения. В нем будет рассмотрена задача о создании систем, которые могли бы приспосабливаться к окружающей среде, а также обучаться на основе получаемого опыта. Такие задачи возникают во многих областях, включая информатику, технические науки, математику, физику, нейробиологию и когнитологию. В середине 2010-х годов методы обучения с подкреплением удалось эффективно применить для обучения глубоких нейронных сетей, что привело к ряду значимых результатов. В рамках спецкурса будут изложены основные методы обучения с подкреплением, приведены техники их успешного использования для глубоких нейронных сетей, рассмотрены примеры, предложены практические задания.

### Слайды с лекций

[Лекция 1. Введение в обучение с подкреплением. Метод Cross-Entropy](https://github.com/imm-rl-lab/UrFU_course/blob/master/Slides/Lecture_1.pdf)

[Лекция 2. Введение в нейронные сети. Deep Cross-Entropy Method](https://github.com/imm-rl-lab/UrFU_course/blob/master/Slides/Lecture_2.pdf)

[Лекция 3. Динамическое программирование](https://github.com/imm-rl-lab/UrFU_course/blob/master/Slides/Lecture_3.pdf)

[Лекция 4. Model-Free Reinforcement Learning](https://github.com/imm-rl-lab/UrFU_course/blob/master/Slides/Lecture_4.pdf)

### Код с практик

[Практика 1. Метод Cross-Entropy для решение Maze](https://github.com/imm-rl-lab/UrFU_course/blob/master/Coding/Practice_1.py)

[Практика 2.1. Решение задачи регрессии с использованием PyTorch](https://github.com/imm-rl-lab/UrFU_course/blob/master/Coding/Practice-2_Problem-1.py)

[Практика 2.2. Решение Cartpole методом Deep Cross-Entropy](https://github.com/imm-rl-lab/UrFU_course/blob/master/Coding/Practice-2_Problem-2.py)

[Практика 3. Решение Frozen Lake методами Policy Iteration и Value Iteration](https://github.com/imm-rl-lab/UrFU_course/blob/master/Coding/Practice-3.py)

### Домашние задания
[Домашнее задание 1](https://github.com/imm-rl-lab/UrFU_course/blob/master/Homework/Homework_1.pdf)

[Домашнее задание 2](https://github.com/imm-rl-lab/UrFU_course/blob/master/Homework/Homework_2.pdf)

[Домашнее задание 3](https://github.com/imm-rl-lab/UrFU_course/blob/master/Homework/Homework_3.pdf)

### Полезные ссылки

[https://gym.openai.com/](https://gym.openai.com/) Страничка библиотеки Gym для Python. В ней содержаться многие стандартные Environments для обучения с подкреплением.

[https://github.com/MattChanTK/gym-maze](https://github.com/MattChanTK/gym-maze) Репозиторий сред c Maze

[https://pytorch.org/](https://pytorch.org/) Сайт библиотеки PyTorch.

[https://playground.tensorflow.org/](https://playground.tensorflow.org/) Страничка с хорошей визуализацией обучения нейронных сетей. Просто так :)

### Видео лекции

[A. Panin. Cross-Entropy Method.](https://ru.coursera.org/lecture/practical-rl/crossentropy-method-TAT8g) Короткая, но понятная лекция по применению метода Cross-Entropy к задачам обучения с подкреплением.

### Литература

[S. Mannor, R. Rubinstein, Y. Gat. The Cross-Entropy method for Fast Policy Search (2003).](https://www.aaai.org/Papers/ICML/2003/ICML03-068.pdf) Статья про использование метода Cross-Entropy для оптимизации Policy в задачах обучения с подкреплением.

[A. Costa, O. Jones, D. Kroese. Convergence properties of the cross-entropy method for discrete optimization (2007)](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.399.4581&rep=rep1&type=pdf) В статье дается обоснование сходимости метода Cross-Entropy в задачах дискретной оптимизации. Однако, если пространство состояний и действий конечные, а среда детерминирована, то, кажется, задача Reinforcement Learning в рассматриваемую постановку задачи дискретной оптимизации вкладывается.

[G. Cybenko. Approximation by Superpositions of a Sigmoidal Function (1989).](https://pdfs.semanticscholar.org/05ce/b32839c26c8d2cb38d5529cf7720a68c3fab.pdf) Теорема Цыбенко об аппроксимации непрерывных функций суперпозициями сигмоидальных функций (считай нейронными сетями).

[C. Николенко, А. Кадурин, Е. Архангельская. Глубокое обучение. Погружение в мир нейронных сетей (2018).](https://cloud.mail.ru/public/AaZw/UM3d856gy) Пожалуй, единственная книга на русском, в которой последовательно и достаточно полно изложены основные моменты работы с нейронными сетями. Написана простым языком, но при этом включает в себя серьёзный обзор литературы со ссылками на первоисточники. 

[Р.С. Саттон, Э.Г. Барто. Обучение с подкреплением (1998).](https://nashol.com/2017091096341/obuchenie-s-podkrepleniem-satton-r-s-barto-e-g-2014.html) Уже ставшая классической монография по обучению с подкреплением.
