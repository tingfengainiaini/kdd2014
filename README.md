kdd2014
=======

数据说明
=======

* donations.csv: 含有每个项目的donations信息。只有训练集有
* essays.csv: 教师提交的项目的文本。训练、测试都有
* projects.csv: 关于每个项目的信息。训练测试都有
* resources.csv: 含有每个项目需要的信息。训练测试都有
* outcomes.csv:训练集的结果
* sampleSubmission.csv:样例输出

"Exciting" Projects
===================

Exciting projects 符合DC.org网站的各种需求。注意，“exciting” 意味着有商业贡献并且这并不意味着non-exciting project会对教师、学生、捐赠者产生影响(are not compelling to teachers/students/donors-这句我不会翻译)。为了成为exciting， 一个项目必须符合全部五个限制。下面是这五个限制，带有括号的是其在数据中的出现名称。

* 被成功完成(fully_funded)
* 有至少一个teacher-acquired donor(at\_least\_1\_teacher\_referred_donor)
* 有超过平均段的捐赠者留言 (great_chat)
* 至少有一个"green"捐赠 (at\_least\_1\_green_donation)
* 有一个或者多个
  - 3个或更多非teacher-acquired捐赠者(three\_or\_more\_non\_teacher\_referred\_donors)
  - 一个非teacher-acquired教师给了超过$100 (one\_non\_teacher\_referred\_donor\_giving\_100\_plus)
  - 项目受到了"thoughtful donor"的捐赠(donation\_from\_thoughtful\_donor)

你将会在outcomes.csv中找到摘要信息，包括is\_exciting变量的真值信息

Data fields
============

下面是提供的数据的一个简短解释。有一些很清楚的就不细说了

* outcomes.csv
  - is\_exciting - 从商业的角度看，是否一个项目是exciting的
  - at\_least\_1\_teacher\_referred\_donor - teacher referred = donor donated because teacher shared a link or publicized their page
  - fully\_funded - 项目成功完成
  - at\_\least\_1\_green\_donation - 由一些大公司(Amazon)或者先进支付的捐赠
  - great\_chat - 项目评论数超过平均
  - three\_or\_more\_non\_teacher\_referred\_donors - non-teacher referred is a donor that landed on the site by means other than a teacher referral link/page
  - one\_non\_teacher\_referred\_donor\_giving\_100\_plus - see above
  - donation\_from\_thoughtful\_donor - 就是一帮很刁的捐赠者
  - great\_messages\_proportion -  great\_chat 如何被算出。 如果大于现阶段的62%，great\_chat=True
  - teacher\_referred\_count - teacher referred的数量 (see above)
  - non\_teacher\_referred\_count - non-teacher referred的数量 (see above)

* projects.csv
  - projectid - 项目id
  - teacher\_acctid - 创建项目的教师的id
  - schoolid - 教师工作的学校的id
  - school\_ncesid - public National Center for Ed Statistics id
  - school\_latitude - 学校维度
  - school\_longitude - 学校经度
  - school\_city - 学校城市
  - school\_state 
  - school\_zip
  - school\_metro - 地铁
  - school\_district - 学校地区
  - school\_county
  - school\_charter - “特许”学校-网上查的
  - school\_magnet - whether a public magnet school or not
  - school\_year\_round - whether a public year round school or not
  - school\_nlns - whether a public nlns school or not
  - school\_kipp - whether a public kipp school or not
  - school\_charter\_ready\_promise - whether a public ready promise school or not
  - teacher\_prefix - 教师性别
  - teacher\_teach\_for\_america - Teach for America or not
  - teacher\_ny\_teaching\_fellow - New York teaching fellow or not
  - primary\_focus\_subject - main subject for which project materials are intended
  - primary\_focus\_area - main subject area for which project materials are intended
  - secondary\_focus\_subject - secondary subject
  - secondary\_focus\_area - secondary subject area
  - resourc\_type - main type of resources requested by a project
  - poverty\_level - school's poverty level. highest: 65%+ free of reduced lunch  high: 40-64% moderate: 10-39% low: 0-9%
  - grade\_level - grade level for which project materials are intended
  - fulfillment\_labor\_materials - cost of fulfillment
  - total\_price\_excluding\_optional\_support - project cost excluding optional tip that donors give to DonorsChoose.org while funding a project
  - total\_price\_including\_optional\_support - see above
  - students\_reached - number of students impacted by a project (if funded)
  - eligible\_double\_your\_impact\_match - project was eligible for a 50% off offer by a corporate partner (logo appears on a project, like Starbucks or Disney)
  - eligible\_almost\_home\_match - project was eligible for a $100 boost offer by a corporate partner
  - date\_posted - data a project went live on the site

下面还有很多，今天先拉到吧


