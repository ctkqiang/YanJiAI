# 眼迹AI
眼迹AI是一套集成了面部识别和物体检测功能的系统，它能够实时监控环境并在检测到危险情况时自动向服务器发送警报，从而迅速通知当局应对风险。对于司机而言，时常面临乘客可能实施抢劫或其他不当行为的潜在威胁。通过在您的手机或行车记录仪上部署这一AI系统，不仅能提供即时的安全保障，还能在关键时刻为您节省宝贵的求助时间，确保您能在第一时间获得必要的帮助。

### 个人授权许可证
```markdown
版权所有 2023至2050

特此授予任何获得眼迹AI应用程序（以下简称“软件”）副本的人免费许可，可根据以下条件使用软件：

- 使用者被允许复制、修改、合并、出版发行、散布、再授权和/或销售本软件的副本。
- 在使用、复制、修改和分发软件的副本时，使用者必须在显著位置保留原始许可声明，包括对眼迹AI应用程序的适当署名，
  并特别标明原作者的姓名为钟智强。
- 在使用者派生的作品中，如果使用了本软件的代码或借鉴了本软件的思想，使用者必须在相关代码、文档或其他材料中明确
  指出眼迹AI应用程序及其对应的贡献，并提供眼迹AI应用程序原作者钟智强的适当署名。
- 使用者不得将眼迹AI应用程序标记为自己的作品，或以任何方式暗示眼迹AI应用程序对派生作品的认可或支持。
- 如果使用者希望在本软件的基础上进行盈利或生产产品，必须获得原作者钟智强的书面许可。
- 眼迹AI应用程序提供的是按"原样"提供的，不提供任何明示或暗示的担保或条件，包括但不限于对适销性、特定用途适用性
  和非侵权性的担保或条件。在任何情况下，眼迹AI应用程序的作者或版权持有人均不承担因使用或无法使用本软件所引起的
  任何索赔、损害或其他责任。
- 眼迹AI应用程序的作者或版权持有人不对因使用、复制、修改、合并、出版发行、散布、再授权和/或销售本软件而产生的
  任何索赔、损害或其他责任承担责任，无论是合同责任、侵权行为或其他原因，即使事先已被告知发生此类损害的可能性。 
  
  以上许可条款和限制适用于使用眼迹AI应用程序的全部或部分功能。使用眼迹AI应用程序即表示接受本许可证的条款和条件。
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

以下是程序演示的示例，以及输出结果:

<br />
<img src="assets/截图/截屏2024-05-23 下午4.01.13.png" style="height: auto !important; width: auto !important;"> 
<br />
<img src="assets/截图/截屏2024-05-23 下午4.01.24.png" style="height: auto !important; width: auto !important;">
<br />
<img src="assets/截图/截屏2024-05-23 下午5.20.57.png" style="height: auto !important; width: auto !important;">

<br />
当应用程序运行时，终端将返回以下信息：

当用户眼睛专注时：
```bash
眼迹AI |> [19:00:07] 性别: (男性: 99.84%) | 情绪: 中性 91.0% | [左眼: 无, 右眼: 无] X: 589, Y: 212, W: 306, H: 306
眼迹AI |> [19:00:07] 性别: (男性: 99.84%) | 情绪: 中性 91.0% | [左眼: 无, 右眼: 无] X: 589, Y: 212, W: 306, H: 306
眼迹AI |> [19:00:07] 性别: (男性: 92.46%) | 情绪: 中性 93.0% | [左眼: 无, 右眼: 无] X: 574, Y: 219, W: 302, H: 302
```

当用户眼睛没有看向前方时：
```bash
[警告：请保持注意力集中在前方!!!] 
```

### 个人捐赠支持
如果您认为该项目对您有所帮助，并且愿意个人捐赠以支持其持续发展和维护，🥰我非常感激您的慷慨。
您的捐赠将帮助我继续改进和添加新功能到该项目中。 通过财务捐赠，您将有助于确保该项目保持免
费和对所有人开放。即使是一小笔捐款也能产生巨大的影响，也是对我个人的鼓励。

## 支持项目 (Support the Project)

如果您认为该项目对您有所帮助，您可以通过以下方式支持项目的持续发展：

### 💰 捐赠渠道 (Donation Channels)

<table>
  <tr>
    <td align="center"><strong>微信支付<br/>WeChat Pay</strong></td>
    <td align="center"><strong>支付宝<br/>Alipay</strong></td>
  </tr>
  <tr>
    <td><img src="https://github.com/ctkqiang/ctkqiang/blob/main/assets/IMG_9863.jpg?raw=true" width="250" /></td>
    <td><img src="https://github.com/ctkqiang/ctkqiang/blob/main/assets/IMG_9859.JPG?raw=true" width="250" /></td>
  </tr>
</table>

### 🌏 国际支付方式 (International Payment Methods)

<div align="center">
  
[![Donate with Ko-fi](https://img.shields.io/badge/Donate_with-Ko--fi-F16061?style=for-the-badge&logo=ko-fi&logoColor=white)](https://ko-fi.com/F1F5VCZJU)
[![Donate with PayPal](https://img.shields.io/badge/Donate_with-PayPal-00457C?style=for-the-badge&logo=paypal&logoColor=white)](https://www.paypal.com/paypalme/ctkqiang)
[![Donate with Stripe](https://img.shields.io/badge/Donate_with-Stripe-626CD9?style=for-the-badge&logo=Stripe&logoColor=white)](https://donate.stripe.com/00gg2nefu6TK1LqeUY)

</div>

您的支持将帮助我:

- 🚀 持续改进项目功能
- 🛠 维护和更新代码
- 📚 提供更好的文档
- 💡 开发新的特性

即使是小额捐赠也能带来巨大帮助。感谢您的支持！ 🙏


## 关注我
<a href="https://twitch.tv/ctkqiang"><img src="https://img.shields.io/badge/Twitch-9146FF?style=for-the-badge&logo=twitch&logoColor=white"></a> <a href="https://open.spotify.com/user/22sblyn4dsymya3xinw3umhai"><img src="https://img.shields.io/badge/Spotify-1ED760?&style=for-the-badge&logo=spotify&logoColor=white"></a> <a href="https://www.tiktok.com/@ctkqiang"><img src="https://img.shields.io/badge/TikTok-000000?style=for-the-badge&logo=tiktok&logoColor=white"></a> <a href="https://stackoverflow.com/users/10758321/%e9%92%9f%e6%99%ba%e5%bc%ba"><img src="https://img.shields.io/badge/Stack_Overflow-FE7A16?style=for-the-badge&logo=stack-overflow&logoColor=white"></a> <a href="https://www.facebook.com/JohnMelodyme/"><img src="https://img.shields.io/badge/Facebook-1877F2?style=for-the-badge&logo=facebook&logoColor=white"></a> <a href="https://github.com/ctkqiang"><img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white"></a> <a href="https://www.instagram.com/ctkqiang"><img src="https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white"></a> <a href="https://www.linkedin.com/in/ctkqiang/"><img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white"></a> <a href="https://linktr.ee/ctkqiang.official"><img src="https://img.shields.io/badge/linktree-39E09B?style=for-the-badge&logo=linktree&logoColor=white"></a> <a href="https://github.com/ctkqiang/ctkqiang/blob/main/assets/IMG_9245.JPG?raw=true"><img src="https://img.shields.io/badge/WeChat-07C160?style=for-the-badge&logo=wechat&logoColor=white"></a>
