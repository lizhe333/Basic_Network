git add .

$commitMessage = Read-Host "请输入commit message"

if ([string]::IsNullOrWhiteSpace($commitMessage)) {
    Write-Host "commit message不能为空，提交取消。"
    exit 1
}

git commit -m $commitMessage
Write-Host "提交成功！"