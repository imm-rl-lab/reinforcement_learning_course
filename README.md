# Страничка курса "Обучение с подкреплением и нейронные сети"

Курс посвящен методам обучения с подкреплением (Reinforcement learning) - одному из способов машинного обучения. В нем будет рассмотрена задача о создании систем, которые могли бы приспосабливаться к окружающей среде, а также обучаться на основе получаемого опыта. Такие задачи возникают во многих областях, включая информатику, технические науки, математику, физику, нейробиологию и когнитологию. В середине 2010-х годов методы обучения с подкреплением удалось эффективно применить для обучения глубоких нейронных сетей, что привело к ряду значимых результатов. В рамках спецкурса будут изложены основные методы обучения с подкреплением, приведены техники их успешного использования для глубоких нейронных сетей, рассмотрены примеры, предложены практические задания.

### Слайды с лекций

Лекция 1. Введение в обучение с подкреплением. Метод Cross-Entropy ([слайды](https://github.com/imm-rl-lab/UrFU_course/blob/master/Slides/Lecture_1.pdf)/[видео](https://www.dropbox.com/s/h2lff3q4rhpzue7/Video_1.mp4?dl=0))

Лекция 2. Введение в нейронные сети. Deep Cross-Entropy Method ([слайды](https://github.com/imm-rl-lab/UrFU_course/blob/master/Slides/Lecture_2.pdf)/[видео](https://www.dropbox.com/s/th4mdrk1jcq1sgx/Video_2.mp4?dl=0))

Лекция 3. Динамическое программирование ([слайды](https://github.com/imm-rl-lab/UrFU_course/blob/master/Slides/Lecture_3.pdf)/[видео](https://www.dropbox.com/s/xipiqohh3zb1o6f/Video_4.mp4?dl=0))

Лекция 4. Model-Free Reinforcement Learning ([слайды](https://github.com/imm-rl-lab/UrFU_course/blob/master/Slides/Lecture_4.pdf))

Лекция 5. Value Function Approximation ([слайды](https://github.com/imm-rl-lab/UrFU_course/blob/master/Slides/Lecture_5.pdf))

Лекция 6. Policy Gradient ([слайды](https://github.com/imm-rl-lab/UrFU_course/blob/master/Slides/Lecture_6.pdf))

### Код с практик

Практика 1. Метод Cross-Entropy для решение Maze ([код](https://github.com/imm-rl-lab/UrFU_course/blob/master/Coding/Practice_1.py))

Практика 2. PyTorch и Deep Cross-Entropy ([код 1](https://github.com/imm-rl-lab/UrFU_course/blob/master/Coding/Practice-2_1.py)/[код 2](https://github.com/imm-rl-lab/UrFU_course/blob/master/Coding/Practice-2_2.py)/[код 3](https://github.com/imm-rl-lab/UrFU_course/blob/master/Coding/Practice-2_3.py)/[видео](https://www.dropbox.com/s/r73q2fowgxgz7yc/Video_3.mp4?dl=0))

Практика 3. Решение Frozen Lake методами Policy Iteration и Value Iteration ([код](https://github.com/imm-rl-lab/UrFU_course/blob/master/Coding/Practice-3.py))

Практика 4. Решение Taxi методами Monte-Carlo, SARSA и Q-Learning ([код](https://github.com/imm-rl-lab/UrFU_course/blob/master/Coding/Practice-4.py))

Практика 5. Решение Cartpole методом DQN ([код](https://github.com/imm-rl-lab/UrFU_course/blob/master/Coding/Practice-5.py))

### Домашние задания
[Домашнее задание 1](https://github.com/imm-rl-lab/UrFU_course/blob/master/Homework/Homework_1.pdf)

[Домашнее задание 2](https://github.com/imm-rl-lab/UrFU_course/blob/master/Homework/Homework_2.pdf)

[Домашнее задание 3](https://github.com/imm-rl-lab/UrFU_course/blob/master/Homework/Homework_3.pdf)

[Домашнее задание 4](https://github.com/imm-rl-lab/UrFU_course/blob/master/Homework/Homework_4.pdf)

### Полезные ссылки

[https://gym.openai.com/](https://gym.openai.com/) Страничка библиотеки Gym для Python. В ней содержаться многие стандартные Environments для обучения с подкреплением.

[https://github.com/MattChanTK/gym-maze](https://github.com/MattChanTK/gym-maze) Репозиторий сред c Maze

[https://pytorch.org/](https://pytorch.org/) Сайт библиотеки PyTorch.

[https://playground.tensorflow.org/](https://playground.tensorflow.org/) Страничка с хорошей визуализацией обучения нейронных сетей. Просто так :)

### Видео лекции

[A. Panin. Cross-Entropy Method.](https://ru.coursera.org/lecture/practical-rl/crossentropy-method-TAT8g) Короткая, но понятная лекция по применению метода Cross-Entropy к задачам обучения с подкреплением.

[D. Silver. Introduction to Reinforcement Learning.](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ) Курс по Reinforcement Learning в University College London.

### Литература

[Р.С. Саттон, Э.Г. Барто. Обучение с подкреплением (1998).](https://nashol.com/2017091096341/obuchenie-s-podkrepleniem-satton-r-s-barto-e-g-2014.html) Уже ставшая классической монография по обучению с подкреплением.

[C. Николенко, А. Кадурин, Е. Архангельская. Глубокое обучение. Погружение в мир нейронных сетей (2018).](https://cloud.mail.ru/public/AaZw/UM3d856gy) Пожалуй, единственная книга на русском, в которой последовательно и достаточно полно изложены основные моменты работы с нейронными сетями. Написана простым языком, но при этом включает в себя серьёзный обзор литературы со ссылками на первоисточники. 

[S. Mannor, R. Rubinstein, Y. Gat. The Cross-Entropy method for Fast Policy Search (2003).](https://www.aaai.org/Papers/ICML/2003/ICML03-068.pdf) Статья про использование метода Cross-Entropy для оптимизации Policy в задачах обучения с подкреплением.

[A. Costa, O. Jones, D. Kroese. Convergence properties of the cross-entropy method for discrete optimization (2007)](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.399.4581&rep=rep1&type=pdf) В статье дается обоснование сходимости метода Cross-Entropy в задачах дискретной оптимизации. Однако, если пространство состояний и действий конечные, а среда детерминирована, то, кажется, задача Reinforcement Learning в рассматриваемую постановку задачи дискретной оптимизации вкладывается.

[G. Cybenko. Approximation by Superpositions of a Sigmoidal Function (1989).](https://pdfs.semanticscholar.org/05ce/b32839c26c8d2cb38d5529cf7720a68c3fab.pdf) Теорема Цыбенко об аппроксимации непрерывных функций суперпозициями сигмоидальных функций (считай нейронными сетями).

[V. Mnih at el. Playing Atari with Deep Reinforcement Learning (2013).](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) Статья про алгоритм DQN в приложении к играм Atari.

[H. Van Hasselt, A. Guez, D. Silver. Deep Reinforcement Learning with Double Q-Learning (2016).](https://arxiv.org/pdf/1509.06461.pdf) Статья про алгоритм Double DQN.

[S. Gu, T. Lillicrap, I. Sutskever, S. Levine. Continuous Deep Q-Learning with Model-based Acceleration (2016).](http://proceedings.mlr.press/v48/gu16.pdf) Статья про алгоритм Continuous DQN.

[D. Silver at el. Deterministic Policy Gradient Algorithms David (2014).](http://proceedings.mlr.press/v32/silver14.pdf) Статья, в которой доказывается Deterministic Policy Gradient Theorem и приводится Deterministic Policy Gradient Algorithm.

[T. Lillicrap at el. Continuous control with deep reinforcement learning (2016)](https://arxiv.org/pdf/1509.02971.pdf) Статья про алгоритм DDPG.

[V. Mnih at el. Asynchronous Methods for Deep Reinforcement Learning (2016).](https://arxiv.org/pdf/1602.01783.pdf) Статья про асинхронный подход для решения задач Reinforcement Learning.


