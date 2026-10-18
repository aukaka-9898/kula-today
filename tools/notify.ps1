# Kula Today notifier (Sam, 2026-10-17: "Scheduled task could notify me,
# 'hey, something new in Kula!'"). Run by a Windows scheduled task at logon
# and once a day. Reads the monthly rule check's report from the published
# page and shows a desktop notification when it has events not seen before.
# Nothing is sent anywhere; the only network call is reading the report.

$ErrorActionPreference = "Stop"
$report = "https://aukaka-9898.github.io/kula-today/monthly_report.json"
$link = "https://github.com/aukaka-9898/kula-today/blob/main/docs/monthly_report.md"
$state = Join-Path $env:LOCALAPPDATA "KulaToday\seen.txt"

try {
    $r = Invoke-RestMethod -Uri ($report + "?ts=" + [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()) -TimeoutSec 30
} catch { exit 0 }                                   # offline: try again next time
if (-not $r.events -or $r.events.Count -eq 0) { exit 0 }

$key = "$($r.month) $($r.made)"
New-Item -ItemType Directory -Force -Path (Split-Path $state) | Out-Null
if ((Test-Path $state) -and ((Get-Content $state -Raw).Trim() -eq $key)) { exit 0 }

[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] | Out-Null
$first = [Security.SecurityElement]::Escape([string]$r.events[0])
$more = if ($r.events.Count -gt 1) { " (+$($r.events.Count - 1) more)" } else { "" }
$xml = New-Object Windows.Data.Xml.Dom.XmlDocument
$xml.LoadXml(@"
<toast activationType="protocol" launch="$link">
  <visual><binding template="ToastGeneric">
    <text>Hey, something new in Kula!</text>
    <text>$first$more</text>
  </binding></visual>
</toast>
"@)
$app = '{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe'
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($app).Show(
    [Windows.UI.Notifications.ToastNotification]::new($xml))
Set-Content -Path $state -Value $key
