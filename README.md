# titanic_kaggle

**1. 문제 정의**

타이타닉 탑승자 데이터를 기반으로 생존자를 예측하는 딥러닝 모델을
생성합니다.

1)  순서

-   Load Dataset(데이터 불러오기)

-   EDA(Exploratory Data Analysis) 탐색적 데이터 분석

-   데이터 전처리 및 특성 추출

-   모델 설계 및 학습

-   검증

2)  탐색적 데이터 분석을 할 때 데이터 시각화를 병행합니다. 파이썬의
    데이터 시각화 패키지 중 matplotlib, seaborn을 사용할 것입니다.

3)  컬럼설명

-   Survival: 생존 여부. 0이면 사망, 1이면 생존한 것으로 간주합니다.

-   Pclass: 티켓 등급. 1등석(1), 2등석(2), 3등석(3)이 있으며,
    1등석일수록 좋고 3등석일수록 좋지 않습니다.

-   Sex: 성별. 남자(male)와 여자(female)이 있습니다.

-   Age: 나이입니다. 틈틈히 빈 값이 존재하며, 소수점 값도 존재합니다.

-   SibSp: 해당 승객과 같이 탑승한 형재/자매(siblings)와
    배우자(spouses)의 총 인원 수입니다.

-   Parch: 해당 승객과 같이 탑승한 부모(parents)와 자식(children)의 총
    인원 수입니다.

-   Ticket: 티켓 번호입니다. 다양한 텍스트(문자열)로 구성되어 있습니다.

-   Fare: 운임 요금입니다. 소수점으로 구성되어 있습니다.

-   Cabin: 객실 번호입니다. 많은 빈 값이 존재하며, 다양한
    텍스트(문자열)로 구성되어 있습니다.

-   Embarked: 선착장입니다. C는 셰르부르(Cherbourg)라는 프랑스 지역, Q는
    퀸스타운(Queenstown)이라는 영국 지역, S는
    사우스햄튼(Southampton)이라는 영국 지역입니다.

**3. 데이터 가공, Feature 선택, 적용모델 설명**

1\) 필요한 파이썬 라이브러리 import

-   데이터 분석용 라이브러리 numpy와 pandas를 import

-   파이썬의 데이터 시각화 패키지 matplotlib과 seaborn을 import

[2) Load Dataset]{.underline}

-   모든 데이터 분석의 시작은 주어진 데이터를 읽어오는 것입니다.

-   \[판다스(Pandas)\](https://pandas.pydata.org/)에는 \[read_csv\]

-   (https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html)라는
    기능이 있는데, 이 기능을 통해 편리하게 데이터를 읽어올 수 있습니다.

-   read_csv를 활용해 \[Titanic: Machine Learning from
    Disaster\](https://www.kaggle.com/c/titanic/) 경진대회에서 제공하는
    두 개의 데이터(train, test)를 읽어오겠습니다.

[2) 데이터 파악]{.underline}

-   pandas의 info( ) 메서드를 사용하여 각 컬럼들의 데이터 성격
    파악합니다.

    -   891개의 관측 데이터와 12개의 컬럼으로 이루어져 있는 것으로
        파악됩니다.

-   Age는177개, Cabin은 687개, Embarked는 2개의 관측치를 가지고
    있습니다.

-   Age는 평균값으로 결측치를 채우고, Cabin은 N으로 채웁니다. Embarked
    또한 N으로 결측치를 채웁니다.

[3) 데이터 분포 확인하기]{.underline}

-   Sex 값 분포 확인

-   Cabin 값 분포 확인

-   Embarked 값 분포 확인

-   Cabin 값 추출: Cabin의 선실 번호 중 선실 등급을 나타내는 첫 번째
    알파벳이 중요하므로 앞 문자만 추출. 등급이 중요한 이유는 부자와
    가난한 사람들간의 등급은 다를 것으로 가정

-   Sex(성별) 이 생존 확률에 어떤 영향을 미쳤는지 시각화하여 확인

-   객실 등급(Pclass)가 생존 확률에 어떤 영향을 미쳤는지 시각화하여
    확인. 객실 등급은 크게 1등급(=퍼스트 클래스), 2등급(=비즈니스),
    3등급(=이코노미) 로 나뉩니다. 추측컨데 객실 등급이 높을 수록 더 많은
    비용을 지불한 타이타닉호의 VIP라고 볼 수 있음

-   연령대에 따른 생존여부는?

  -   나이를 가지고 사용자 함수를 작성하여 연령대라는 범주형 데이터를 생성

  -   Age 값이 너무 다양하기 때문에 범위별로 분류해서 값을 할당

    -   Baby : 0\~5세

    -   Child: 6\~12세

    -   Teenager: 13\~18세

    -   Student: 19\~25세

    -   Young Adult: 26\~35세

    -   Adult: 36\~60세

    -   Elderly: 61세 이상

    -   Unknown: -1이하의 오류 값

-   연령대별 생존 여부를 그래프로 그려 내용을 확인

-   입력 age에 따라 구분 값을 반환하는 함수 설정. DataFrame의 apply
    lambda식에 사용

-   lambda 식에 위에서 생성한 get_category( ) 함수를 반환 값으로 지정

-   get_category(X)는 입력 값으로 \'Age\' 컬럼 값을 받아서 해당하는 cat
    반환

[4) 데이터 인코딩 & 전처리]{.underline}

-   범주형 데이터의 경우 분석에 바로 사용할 수 없기에 원핫인코더,
    라벨인코더로 변경합니다.

-   이번에는 라벨인코더를 사용

```{=html}
<!-- -->
```
-   Null 값 및 불필요한 컬럼 삭제

-   레이블 인코딩 수행

-   transform_features( ) 함수를 호출

[5)생존 결과 예측]{.underline}

-   원본 데이터를 재로딩하고 타이타닉 생존자 데이터 셋의 레이블인
    Survivied 속성만 별도 분리해 데이터 셋으로 추출

-   학습 데이터를 기반으로 하여 train_test_split( )으로 별도의 데이터 셋
    추출

[6) 데이터 스케일링]{.underline}

-   딥러닝은 스케일링을 하지 않으면 숫자가 컬럼이 의미가 커져벼러서
    제대로 된 학습이 되지 않으므로 딥러닝전에 스케일링을 진행

-   MinMaxScaler 변환기로 최대/최소값이 각각 1,0이 되도록 조정

[7) 모델 생성 및 학습하기]{.underline}

-   텐서플로우를 import

-   Keras의 Sequential 을 이용해 모델링

```{=html}
<!-- -->
```
-   케라스 모델중 순차형 모델 적용, 함수형 모델은 전이학습 등 복잡한
    모델을 설계할 때 사용하므로 이번 과제에서는 굳이 함수형을 사용할
    필요 없음

```{=html}
<!-- -->
```
-   생존자 사망자 분류는 생존 or 사망으로 이진 분류이므로 Sigmoid 함수를
    활성화 함수로 사용

-   Compile(모델 컴파일)

```{=html}
<!-- -->
```
-   Survived 변수는 가질 수 있는 값이 0과 1, 두 가지밖에 없는 이분형
    변수이므로 loss를 binary_crossentropy로 사용

-   Adam과 Adagrad(Adative Gradient) 둘 다 돌렸을 때 Adagrad가 더 좋은
    성능을 내서 Adagrad로 선택

```{=html}
<!-- -->
```
-   epochs은 학습 데이터를 한번 전체 다 소모할 때 하나의 epoch이라고
    하는데 500번 학습시켰지만 callback 함수로 earlystop 로직을 걸어서
    학습에 변화가 없으면 학습을 멈춤

-   loss는 오버피팅을 확인하기 위함

[8) 테스트 데이터 검증]{.underline}

-   타이타닉의 경우 평가방법이 Accuracy(정확도)이므로 스코어 예측을 제출
    전 확인

[9) Submit]{.underline}

-   캐글의 타이타닉 경진대회에서는 gender_submission.csv라는 제출 포맷을
    제공합니다. 이 제출 포맷에 맞게 집어넣고 저장할 것입니다.

