// Lets the local launcher know the page is still open (used only in browser mode).
const ping = () => fetch("/api/ping", { method: "POST", cache: "no-store" }).catch(() => {});
ping();
setInterval(ping, 10000);
