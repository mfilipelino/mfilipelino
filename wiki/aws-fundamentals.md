---
cover: >-
  https://images.unsplash.com/photo-1628296499994-70face79ab36?crop=entropy&cs=srgb&fm=jpg&ixid=M3wxOTcwMjR8MHwxfHNlYXJjaHwyfHxhd3N8ZW58MHx8fHwxNjk3Mjk1NTEyfDA&ixlib=rb-4.0.3&q=85
coverY: 0
---

# 💳 AWS Fundamentals

### Shared Responsibility Model

Security and Compliance is a shared responsibility between AWS and the customer.

<figure><img src="https://d1.awsstatic.com/security-center/Shared_Responsibility_Model_V2.59d1eccec334b366627e9295b304202faf7b899b.jpg" alt=""><figcaption><p><a href="https://aws.amazon.com/compliance/shared-responsibility-model/">https://aws.amazon.com/compliance/shared-responsibility-model/</a></p></figcaption></figure>

IAM  - Identity and Access management&#x20;

* **Global** service
* **Root** accounts created by default, shouldn't be used or shared
* **Users** are people within your organization and can be grouped
* **Groups** only contain users, not other groups
* Users don't have to belong to a group and users can belong to multiple groups
* A user can belong to **multiple groups**



**IAM Permissions:** Users or Groups can be assigned JSON documents called policies, that allow them to have permission to use some actions in some specific resources

**The policies** define the permissions of the users

AWS explicitly applies the security principle: **Least privilege principle:** don't give more permissions than a user needs











