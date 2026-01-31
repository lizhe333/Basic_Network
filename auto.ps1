git add .

$commitMessage = Read-Host "请输入commit message"

if ([string]::IsNullOrWhiteSpace($commitMessage)) {
    Write-Host "commit message不能为空，提交取消。"
    exit 1
}

git commit -m $commitMessage

# 推送到远程仓库
$branchName = git branch --show-current
Write-Host "正在推送更改到分支: $branchName"
git push origin $branchName

Write-Host "提交并推送成功！"