# ArguminSci
Analyze Argumentation and other Rhetorical Aspects in Scientific Writing.

Here you can find the code necessary to deploy ArguminSci. A detailed description of the tool can be found in the accompanying publication [1].



ArguminSci's models are not included in this Github repository due to space reasons. They can be found here: http://data.dws.informatik.uni-mannheim.de/arguminsci/.
In order to include them, just download the folder and copy it to ArguminSci's root. The final project structure should look like this:

```
Arguminsci
- static (static files for the web app)
- templates (for the template rendering engine used for the web app)
- test (a test file)
- model (all models you want to include)
  - argumentation
  - aspect
  - citation
  - discourse
  - summary
```

The models are all trained on the extended version of the Dr. Inventor Corpus [2,3,4].


Please cite the corresponding publication in case you make use of our tools. 

[1] Anne Lauscher, Goran Glavaˇs, and Kai Eckert. 2018. ArguminSci: A Tool for Analyzing Argumentation and Rhetorical Aspects in Scientific Writing. https://www.aclweb.org/anthology/W18-5203/
In Proceedings of the 5th Workshop on Mining Argumentation, 22–28, Brussels, Belgium. Association for Computational Linguistics.
```
@inproceedings{lauscher2018a,
  title = {ArguminSci: A Tool for Analyzing Argumentation and Rhetorical Aspects in Scientific Writing},
  booktitle = {Proceedings of the 5th Workshop on Mining Argumentation},
  publisher = {Association for Computational Linguistics},
  author = {Lauscher, Anne and Glava\v{s}, Goran and Eckert, Kai},
  address = {Brussels, Belgium},
  year = {2018},
  pages = {22–28}
}
```

[2] Anne Lauscher, Goran Glavaˇs, and Simone Paolo Ponzetto. 2018. An argument-annotated corpus of scientific publications. https://www.aclweb.org/anthology/W18-5206/ 
In Proceedings of the 5th Workshop on Mining Argumentation, 40–46, Brussels, Belgium. Association for Computational Linguistics.
```
@inproceedings{lauscher2018b,
  title = {An argument-annotated corpus of scientific publications},
  booktitle = {Proceedings of the 5th Workshop on Mining Argumentation},
  publisher = {Association for Computational Linguistics},
  author = {Lauscher, Anne and Glava\v{s}, Goran and Ponzetto, Simone Paolo},
  address = {Brussels, Belgium},
  year = {2018},
  pages = {40–46}
}
```

[3] Beatriz Fisas, Francesco Ronzano, and Horacio Saggion.
2016. A multi-layered annotated corpus of
scientific papers. In Proceedings of the International
Conference on Language Resources and Evaluation,
pages 3081–3088, Portoroˇz, Slovenia. European
Language Resources Association.

[4] Beatriz Fisas, Horacio Saggion, and Francesco Ronzano.
2015. On the discoursive structure of computer
graphics research papers. In Proceedings of
The 9th Linguistic Annotation Workshop, pages 42–
51, Denver, CO, USA. Association for Computational
Linguistics.
