# CoCoRec
The official implementation of the CoCoRec algorithm proposed in the paper ["Category-aware Collaborative Sequential Recommendation"](https://dl.acm.org/doi/abs/10.1145/3404835.3462832) which is accepted to SIGIR 2021.

To run the code,
```
      python main_time.py
```

The datasets are Taobao and BeerAdvocate. Basically any dataset having sequences of user actions and the category information of items can be used for the algorithms.

BibTeX for citation:

```
@inproceedings{cai2021category,
  title={Category-aware collaborative sequential recommendation},
  author={Cai, Renqin and Wu, Jibang and San, Aidan and Wang, Chong and Wang, Hongning},
  booktitle={Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={388--397},
  year={2021}
}
```
