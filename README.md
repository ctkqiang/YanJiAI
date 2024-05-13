# 眼迹AI
眼迹AI是一套集成了面部识别和物体检测功能的系统，它能够实时监控环境并在检测到危险情况时自动向服务器发送警报，从而迅速通知当局应对风险。对于司机而言，时常面临乘客可能实施抢劫或其他不当行为的潜在威胁。通过在您的手机或行车记录仪上部署这一AI系统，不仅能提供即时的安全保障，还能在关键时刻为您节省宝贵的求助时间，确保您能在第一时间获得必要的帮助。

### 个人授权许可证
```markdown
版权所有 2023至2050

特此授予任何获得眼迹AI应用程序（以下简称“软件”）副本的人免费许可，可根据以下条件使用软件：

- 使用者被允许复制、修改、合并、出版发行、散布、再授权和/或销售本软件的副本。
- 在使用、复制、修改和分发软件的副本时，使用者必须在显著位置保留原始许可声明，包括对华佗AI应用程序的适当署名，
  并特别标明原作者的姓名为钟智强。
- 在使用者派生的作品中，如果使用了本软件的代码或借鉴了本软件的思想，使用者必须在相关代码、文档或其他材料中明确
  指出华佗AI应用程序及其对应的贡献，并提供华佗AI应用程序原作者钟智强的适当署名。
- 使用者不得将华佗AI应用程序标记为自己的作品，或以任何方式暗示华佗AI应用程序对派生作品的认可或支持。
- 如果使用者希望在本软件的基础上进行盈利或生产产品，必须获得原作者钟智强的书面许可。
- 华佗AI应用程序提供的是按"原样"提供的，不提供任何明示或暗示的担保或条件，包括但不限于对适销性、特定用途适用性
  和非侵权性的担保或条件。在任何情况下，华佗AI应用程序的作者或版权持有人均不承担因使用或无法使用本软件所引起的
  任何索赔、损害或其他责任。
- 华佗AI应用程序的作者或版权持有人不对因使用、复制、修改、合并、出版发行、散布、再授权和/或销售本软件而产生的
  任何索赔、损害或其他责任承担责任，无论是合同责任、侵权行为或其他原因，即使事先已被告知发生此类损害的可能性。 
  
  以上许可条款和限制适用于使用华佗AI应用程序的全部或部分功能。使用华佗AI应用程序即表示接受本许可证的条款和条件。
```


### 配置
```bash
pip3 install -r requirements.txt
```

如果您遇到以下问题 (MacOS)：
```bash
ssl.SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: 
unable to get local issuer certificate
```
请执行以下命令: 
```bash
/Applications/Python\{您的python版本}/Install\ Certificates.command ; exit;

# 目前我的版本是 Python 3.10 所以以下的是我命令:
/Applications/Python\3.10/Install\ Certificates.command ; exit;
```

### 用法

```bash
python3 run.py
```

### 测试 
```bash
python3 -m unittest tests/test.py
```

### 个人捐赠支持
如果您认为该项目对您有所帮助，并且愿意个人捐赠以支持其持续发展和维护，🥰我非常感激您的慷慨。
您的捐赠将帮助我继续改进和添加新功能到该项目中。 通过财务捐赠，您将有助于确保该项目保持免
费和对所有人开放。即使是一小笔捐款也能产生巨大的影响，也是对我个人的鼓励。

以下是我的支付宝二维码，您可以扫描二维码进行个人捐赠：

<br />
<div style="display: flex; justify-content: space-between; margin-bottom: 20px;">
  <img src="https://github.com/ctkqiang/ctkqiang/blob/main/assets/IMG_9863.jpg?raw=true" style="height: 500px !important; width: 350px !important;">
 
  <img src="https://github.com/ctkqiang/ctkqiang/blob/main/assets/IMG_9859.JPG?raw=true" style="height: 500px !important; width: 350px !important;">
</div>


[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/F1F5VCZJU)



## 爱心捐款
<a href="https://qr.alipay.com/fkx19369scgxdrkv8mxso92"><img src="https://img.shields.io/badge/alipay-00A1E9?style=for-the-badge&logo=alipay&logoColor=white"></a> <a href="https://ko-fi.com/F1F5VCZJU"><img src="https://img.shields.io/badge/Ko--fi-F16061?style=for-the-badge&logo=ko-fi&logoColor=white"></a> <a href="https://www.paypal.com/paypalme/ctkqiang"><img src="https://img.shields.io/badge/PayPal-00457C?style=for-the-badge&logo=paypal&logoColor=white"></a> <a href="https://donate.stripe.com/00gg2nefu6TK1LqeUY"><img src="https://img.shields.io/badge/Stripe-626CD9?style=for-the-badge&logo=Stripe&logoColor=white"></a>

## 关注我
<a href="https://twitch.tv/ctkqiang"><img src="https://img.shields.io/badge/Twitch-9146FF?style=for-the-badge&logo=twitch&logoColor=white"></a> <a href="https://open.spotify.com/user/22sblyn4dsymya3xinw3umhai"><img src="https://img.shields.io/badge/Spotify-1ED760?&style=for-the-badge&logo=spotify&logoColor=white"></a> <a href="https://www.tiktok.com/@ctkqiang"><img src="https://img.shields.io/badge/TikTok-000000?style=for-the-badge&logo=tiktok&logoColor=white"></a> <a href="https://stackoverflow.com/users/10758321/%e9%92%9f%e6%99%ba%e5%bc%ba"><img src="https://img.shields.io/badge/Stack_Overflow-FE7A16?style=for-the-badge&logo=stack-overflow&logoColor=white"></a> <a href="https://www.facebook.com/JohnMelodyme/"><img src="https://img.shields.io/badge/Facebook-1877F2?style=for-the-badge&logo=facebook&logoColor=white"></a> <a href="https://github.com/ctkqiang"><img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white"></a> <a href="https://www.instagram.com/ctkqiang"><img src="https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white"></a> <a href="https://www.linkedin.com/in/ctkqiang/"><img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white"></a> <a href="https://linktr.ee/ctkqiang.official"><img src="https://img.shields.io/badge/linktree-39E09B?style=for-the-badge&logo=linktree&logoColor=white"></a> <a href="https://github.com/ctkqiang/ctkqiang/blob/main/assets/IMG_9245.JPG?raw=true"><img src="https://img.shields.io/badge/WeChat-07C160?style=for-the-badge&logo=wechat&logoColor=white"></a>


 
